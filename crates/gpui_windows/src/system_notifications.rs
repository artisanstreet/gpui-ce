//! System notifications as Windows toast notifications.

use std::cell::RefCell;
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash as _, Hasher as _};
use std::rc::Rc;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::channel::mpsc;
use gpui::{
    ForegroundExecutor, SharedString, SystemNotification, SystemNotificationResponse, Task,
};
use parking_lot::Mutex;
use windows::Data::Xml::Dom::XmlDocument;
use windows::Foundation::TypedEventHandler;
use windows::UI::Notifications::{
    ToastActivatedEventArgs, ToastDismissedEventArgs, ToastFailedEventArgs, ToastNotification,
    ToastNotificationManager, ToastNotifier,
};
use windows::core::HSTRING;
use windows::core::{IInspectable, Interface as _, h};

type ResponseCallback = Rc<RefCell<Option<Box<dyn FnMut(SystemNotificationResponse)>>>>;

pub(crate) struct SystemNotificationState {
    notifier: Option<ToastNotifier>,
    /// The AppUserModelID an unpackaged app's toasts are filed under, which
    /// Notification Center needs to find them again.
    app_identifier: Option<HSTRING>,
    active_toasts: Arc<Mutex<HashMap<SharedString, (u64, ToastNotification)>>>,
    next_generation: u64,
    response_sender: mpsc::UnboundedSender<SystemNotificationResponse>,
    response_receiver: Option<mpsc::UnboundedReceiver<SystemNotificationResponse>>,
    callback: ResponseCallback,
    _response_task: Option<Task<()>>,
}

impl SystemNotificationState {
    pub(crate) fn new() -> Self {
        let (response_sender, response_receiver) = mpsc::unbounded();
        Self {
            notifier: None,
            app_identifier: None,
            active_toasts: Arc::new(Mutex::new(HashMap::new())),
            next_generation: 0,
            response_sender,
            response_receiver: Some(response_receiver),
            callback: Rc::new(RefCell::new(None)),
            _response_task: None,
        }
    }

    pub(crate) fn show(
        &mut self,
        has_package_identity: bool,
        app_identity: Option<(&str, &str)>,
        notification: SystemNotification,
    ) -> windows::core::Result<()> {
        let Some(notifier) = self.notifier(has_package_identity, app_identity)? else {
            return Ok(());
        };

        let document = toast_document(&notification)?;
        let toast = ToastNotification::CreateToastNotification(&document)?;
        let generation = self.next_generation;
        self.next_generation += 1;
        toast.SetTag(&native_tag(&notification.tag))?;
        toast.SetGroup(TOAST_GROUP)?;

        let sender = self.response_sender.clone();
        let response_tag = notification.tag.clone();
        let active_toasts = Arc::downgrade(&self.active_toasts);
        toast.Activated(&TypedEventHandler::<ToastNotification, IInspectable>::new(
            move |_sender, arguments| {
                if let Some(active_toasts) = active_toasts.upgrade() {
                    remove_current_toast(&active_toasts, &response_tag, generation);
                }
                let action_id = arguments
                    .as_ref()
                    .and_then(|arguments| arguments.cast::<ToastActivatedEventArgs>().ok())
                    .and_then(|arguments| arguments.Arguments().ok())
                    .filter(|arguments| !arguments.is_empty())
                    .map(|arguments| SharedString::from(arguments.to_string()));
                sender
                    .unbounded_send(SystemNotificationResponse {
                        tag: response_tag.clone(),
                        action_id,
                    })
                    .ok();
                Ok(())
            },
        ))?;

        let active_toasts = Arc::downgrade(&self.active_toasts);
        let dismissed_tag = notification.tag.clone();
        toast.Dismissed(&TypedEventHandler::<
            ToastNotification,
            ToastDismissedEventArgs,
        >::new(move |_, _| {
            if let Some(active_toasts) = active_toasts.upgrade() {
                remove_current_toast(&active_toasts, &dismissed_tag, generation);
            }
            Ok(())
        }))?;

        let active_toasts = Arc::downgrade(&self.active_toasts);
        let failed_tag = notification.tag.clone();
        toast.Failed(
            &TypedEventHandler::<ToastNotification, ToastFailedEventArgs>::new(move |_, _| {
                if let Some(active_toasts) = active_toasts.upgrade() {
                    remove_current_toast(&active_toasts, &failed_tag, generation);
                }
                Ok(())
            }),
        )?;

        // Never hold the map lock across native calls, which may deliver an event.
        let previous = self.active_toasts.lock().remove(&notification.tag);
        if let Some((_, previous)) = previous {
            notifier.Hide(&previous)?;
        }
        // Register before Show: an immediate native failure/dismissal must be able
        // to remove this entry. Late events from the previous toast use its generation.
        self.active_toasts
            .lock()
            .insert(notification.tag.clone(), (generation, toast.clone()));
        if let Err(error) = notifier.Show(&toast) {
            remove_current_toast(&self.active_toasts, &notification.tag, generation);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn dismiss(&mut self, tag: &str) {
        let Some(notifier) = &self.notifier else {
            return;
        };
        let shown = self.active_toasts.lock().remove(tag);
        if let Some((_, toast)) = shown
            && let Err(error) = notifier.Hide(&toast)
        {
            log::warn!("failed to hide system notification: {error}");
        }
        // Hide only reaches a toast still on screen. One that timed out sits
        // in Notification Center, which badges the taskbar button until it
        // goes, and Windows removes it by tag rather than by object.
        let tag = native_tag(tag);
        let removed = ToastNotificationManager::History().and_then(|history| {
            match &self.app_identifier {
                Some(app_identifier) => {
                    history.RemoveGroupedTagWithId(&tag, TOAST_GROUP, app_identifier)
                }
                None => history.RemoveGroupedTag(&tag, TOAST_GROUP),
            }
        });
        if let Err(error) = removed {
            log::warn!("failed to remove system notification from Notification Center: {error}");
        }
    }

    pub(crate) fn on_response(
        &mut self,
        executor: &ForegroundExecutor,
        callback: Box<dyn FnMut(SystemNotificationResponse)>,
    ) {
        *self.callback.borrow_mut() = Some(callback);

        if let Some(mut receiver) = self.response_receiver.take() {
            let callback = self.callback.clone();
            self._response_task = Some(executor.spawn(async move {
                while let Some(response) = receiver.next().await {
                    // Take the callback out for the call: it may re-enter the
                    // platform (e.g. to dismiss the notification it was told
                    // about) or replace itself.
                    let taken = callback.borrow_mut().take();
                    if let Some(mut taken) = taken {
                        taken(response);
                        callback.borrow_mut().get_or_insert(taken);
                    }
                }
            }));
        }
    }

    fn notifier(
        &mut self,
        has_package_identity: bool,
        app_identity: Option<(&str, &str)>,
    ) -> windows::core::Result<Option<ToastNotifier>> {
        if let Some(notifier) = &self.notifier {
            return Ok(Some(notifier.clone()));
        }

        let notifier = if has_package_identity {
            ToastNotificationManager::CreateToastNotifier()?
        } else {
            let Some((app_identifier, app_name)) = app_identity else {
                log::warn!(
                    "cannot show a system notification without an app identity; \
                     call `App::set_app_identity` during startup"
                );
                return Ok(None);
            };
            register_app_user_model_id(app_identifier, app_name);
            let app_identifier = HSTRING::from(app_identifier);
            let notifier = ToastNotificationManager::CreateToastNotifierWithId(&app_identifier)?;
            self.app_identifier = Some(app_identifier);
            notifier
        };

        self.notifier = Some(notifier.clone());
        Ok(Some(notifier))
    }
}

/// Every toast shares one group, so Notification Center can be asked to
/// remove one by its tag.
const TOAST_GROUP: &HSTRING = h!("gpui");

/// Windows caps toast tags at 64 characters (post-Creators Update), so the
/// arbitrary GPUI tag is hashed down to a fixed-width value.
fn native_tag(tag: &str) -> HSTRING {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    format!("{:016x}", hasher.finish()).into()
}

/// Native events may arrive after the same tag has already been replaced.
fn remove_current_toast<T>(
    toasts: &Mutex<HashMap<SharedString, (u64, T)>>,
    tag: &str,
    generation: u64,
) {
    let removed = {
        let mut toasts = toasts.lock();
        if toasts
            .get(tag)
            .is_some_and(|(current, _)| *current == generation)
        {
            toasts.remove(tag)
        } else {
            None
        }
    };
    // Releasing the COM object can call native code too.
    drop(removed);
}

fn toast_document(notification: &SystemNotification) -> windows::core::Result<XmlDocument> {
    let document = XmlDocument::new()?;
    let toast = document.CreateElement(h!("toast"))?;
    document.AppendChild(&toast)?;

    let visual = document.CreateElement(h!("visual"))?;
    toast.AppendChild(&visual)?;
    let binding = document.CreateElement(h!("binding"))?;
    binding.SetAttribute(h!("template"), h!("ToastGeneric"))?;
    visual.AppendChild(&binding)?;
    for text in [&notification.title, &notification.body] {
        let element = document.CreateElement(h!("text"))?;
        element.SetInnerText(&text.as_ref().into())?;
        binding.AppendChild(&element)?;
    }

    if !notification.actions.is_empty() {
        let actions = document.CreateElement(h!("actions"))?;
        toast.AppendChild(&actions)?;
        for action in &notification.actions {
            let action_element = document.CreateElement(h!("action"))?;
            action_element.SetAttribute(h!("content"), &action.label.as_ref().into())?;
            action_element.SetAttribute(h!("arguments"), &action.id.as_ref().into())?;
            actions.AppendChild(&action_element)?;
        }
    }

    let audio = document.CreateElement(h!("audio"))?;
    audio.SetAttribute(h!("silent"), h!("true"))?;
    toast.AppendChild(&audio)?;
    Ok(document)
}

/// Registers the app's AUMID so toasts display correctly for an unpackaged app without a Start Menu shortcut.
fn register_app_user_model_id(app_identifier: &str, app_name: &str) {
    let result = windows_registry::CURRENT_USER
        .create(format!(r"Software\Classes\AppUserModelId\{app_identifier}"))
        .and_then(|key| key.set_string("DisplayName", app_name));
    if let Err(error) = result {
        log::warn!("failed to register AppUserModelID; notifications may not display: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Weak,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn late_toast_event_does_not_remove_replacement() {
        let toasts = Mutex::new(HashMap::from([("thread".into(), (2, "replacement"))]));
        remove_current_toast(&toasts, "thread", 1);
        assert_eq!(toasts.lock().get("thread"), Some(&(2, "replacement")));

        remove_current_toast(&toasts, "thread", 2);
        assert!(toasts.lock().is_empty());
        remove_current_toast(&toasts, "thread", 2);
        assert!(toasts.lock().is_empty());
    }

    struct DropProbe {
        toasts: Weak<Mutex<HashMap<SharedString, (u64, DropProbe)>>>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            // Native toast release can re-enter an event callback.
            assert!(self.toasts.upgrade().unwrap().try_lock().is_some());
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn finished_toast_is_released_once_outside_the_registry_lock() {
        let toasts = Arc::new(Mutex::new(HashMap::new()));
        let drops = Arc::new(AtomicUsize::new(0));
        toasts.lock().insert(
            "thread".into(),
            (
                7,
                DropProbe {
                    toasts: Arc::downgrade(&toasts),
                    drops: drops.clone(),
                },
            ),
        );

        remove_current_toast(&toasts, "thread", 7);
        remove_current_toast(&toasts, "thread", 7);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
