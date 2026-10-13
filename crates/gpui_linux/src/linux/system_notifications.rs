//! System notifications over the XDG notifications D-Bus interface, via
//! `notify-rust`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use futures::StreamExt as _;
use futures::channel::mpsc;
use gpui::{
    BackgroundExecutor, ForegroundExecutor, SharedString, SystemNotification,
    SystemNotificationResponse, Task,
};
use notify_rust::NotificationResponse;

/// The XDG action key invoked when the user activates the notification body
/// rather than a specific action button.
const DEFAULT_ACTION: &str = "default";

type ResponseCallback = Rc<RefCell<Option<Box<dyn FnMut(SystemNotificationResponse)>>>>;

pub(crate) struct SystemNotificationState {
    response_sender: mpsc::UnboundedSender<SystemNotificationResponse>,
    response_receiver: Option<mpsc::UnboundedReceiver<SystemNotificationResponse>>,
    callback: ResponseCallback,
    _response_task: Option<Task<()>>,
}

impl SystemNotificationState {
    pub(crate) fn new() -> Self {
        let (response_sender, response_receiver) = mpsc::unbounded();
        Self {
            response_sender,
            response_receiver: Some(response_receiver),
            callback: Rc::new(RefCell::new(None)),
            _response_task: None,
        }
    }

    pub(crate) fn show(
        &self,
        executor: &BackgroundExecutor,
        app_name: Option<&str>,
        notification: SystemNotification,
    ) {
        let mut builder = notify_rust::Notification::new();
        if let Some(app_name) = app_name {
            builder.appname(app_name);
        }
        builder
            .summary(&notification.title)
            .body(&notification.body)
            .action(DEFAULT_ACTION, DEFAULT_ACTION);
        let mut action_ids = HashMap::new();
        for (index, action) in notification.actions.iter().enumerate() {
            let transport_id = format!("gpui-action-{index}");
            builder.action(&transport_id, &action.label);
            action_ids.insert(transport_id, action.id.clone());
        }
        let built = builder.finalize();

        let sender = self.response_sender.clone();
        let tag = notification.tag.clone();
        // The D-Bus future sleeps until a response arrives, without retaining a
        // dedicated blocked OS thread and its stack for each live notification.
        executor
            .spawn(async move {
                match built.show_async().await {
                    Ok(handle) => {
                        handle
                            .wait_for_action_async(|response| {
                                if let Some(action_id) = response_action_id(response, &action_ids) {
                                    sender
                                        .unbounded_send(SystemNotificationResponse {
                                            tag,
                                            action_id,
                                        })
                                        .ok();
                                }
                            })
                            .await
                    }
                    Err(error) => log::warn!("failed to show system notification: {error}"),
                }
            })
            .detach();
    }

    pub(crate) fn dismiss(&self, _tag: &str) {
        // Response tasks own the native handles. Dismissal needs a separate
        // tag-to-handle registry so it can target the server-assigned id.
    }

    pub(crate) fn on_response(
        &mut self,
        executor: &ForegroundExecutor,
        callback: Box<dyn FnMut(SystemNotificationResponse)>,
    ) {
        *self.callback.borrow_mut() = Some(callback);

        // Responses arrive from background tasks; this task hands
        // them to the registered callback on the main thread.
        if let Some(mut receiver) = self.response_receiver.take() {
            let callback = self.callback.clone();
            self._response_task = Some(executor.spawn(async move {
                while let Some(response) = receiver.next().await {
                    // Take the callback out for the call: it may re-enter the
                    // platform or replace itself.
                    let taken = callback.borrow_mut().take();
                    if let Some(mut taken) = taken {
                        taken(response);
                        callback.borrow_mut().get_or_insert(taken);
                    }
                }
            }));
        }
    }
}

fn response_action_id(
    response: &NotificationResponse,
    action_ids: &HashMap<String, SharedString>,
) -> Option<Option<SharedString>> {
    match response {
        NotificationResponse::Default => Some(None),
        NotificationResponse::Action(transport_id) => {
            if let Some(action_id) = action_ids.get(transport_id) {
                Some(Some(action_id.clone()))
            } else {
                log::warn!("system notification returned unknown action {transport_id:?}");
                None
            }
        }
        NotificationResponse::Closed(_) | NotificationResponse::Reply(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_action_ids_do_not_collide_with_transport_action_ids() {
        let action_ids = HashMap::from([
            ("gpui-action-0".to_string(), SharedString::from("default")),
            ("gpui-action-1".to_string(), SharedString::from("__closed")),
        ]);

        assert_eq!(
            response_action_id(&NotificationResponse::Default, &action_ids),
            Some(None)
        );
        assert_eq!(
            response_action_id(
                &NotificationResponse::Action("gpui-action-0".into()),
                &action_ids
            ),
            Some(Some("default".into()))
        );
        assert_eq!(
            response_action_id(
                &NotificationResponse::Action("gpui-action-1".into()),
                &action_ids
            ),
            Some(Some("__closed".into()))
        );
        assert_eq!(
            response_action_id(&NotificationResponse::Action("unknown".into()), &action_ids),
            None
        );
    }

    #[test]
    fn closed_or_unsupported_responses_do_not_activate_the_notification() {
        for response in [
            NotificationResponse::Closed(notify_rust::CloseReason::Expired),
            NotificationResponse::Closed(notify_rust::CloseReason::Dismissed),
            NotificationResponse::Closed(notify_rust::CloseReason::CloseAction),
            NotificationResponse::Reply("unsupported on XDG".into()),
        ] {
            assert_eq!(response_action_id(&response, &HashMap::new()), None);
        }
    }
}
