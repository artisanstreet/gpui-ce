#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    println!("cargo:rustc-check-cfg=cfg(gpui_precompiled_shaders)");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var_os("CARGO_FEATURE_WGPU").is_some()
    {
        return Ok(());
    }
    for file in [
        "shaders.hlsl",
        "alpha_correction.hlsl",
        "color_text_raster.hlsl",
        "shaders_mesh.hlsl",
    ] {
        println!("cargo:rerun-if-changed=src/{file}");
    }
    for name in ["GPUI_FXC_PATH", "GPUI_VKD3D_COMPILER", "PATH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    // Native Windows debug builds compile from source at runtime for hot reload.
    // A cross-compiled executable cannot open its build host's source paths.
    if cfg!(debug_assertions) && env::var("HOST")? == env::var("TARGET")? {
        return Ok(());
    }
    let compiler = Compiler::find()?;
    let source_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing manifest directory")?)
            .join("src");
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing output directory")?);
    let use_directwrite = env::var_os("CARGO_FEATURE_COSMIC_TEXT").is_none();
    let scene_modules = [
        "quad",
        "shadow",
        "path_rasterization",
        "path_sprite",
        "underline",
        "monochrome_sprite",
        "subpixel_sprite",
        "polychrome_sprite",
        "blur_downsample",
        "blur",
        "blur_composite",
        "snapshot_transition",
        "emoji_rasterization",
    ]
    .into_iter()
    .filter(|module| *module != "emoji_rasterization" || use_directwrite)
    .map(|module| {
        (
            module,
            if module == "emoji_rasterization" {
                "color_text_raster.hlsl"
            } else {
                "shaders.hlsl"
            },
        )
    })
    .collect::<Vec<_>>();
    compile_modules(
        &compiler,
        &source_dir,
        &output_dir,
        &scene_modules,
        "shaders_bytes.rs",
    )?;
    compile_modules(
        &compiler,
        &source_dir,
        &output_dir,
        &[
            ("mesh", "shaders_mesh.hlsl"),
            ("mesh_composite", "shaders_mesh.hlsl"),
        ],
        "mesh_shaders_bytes.rs",
    )?;
    println!("cargo:rustc-cfg=gpui_precompiled_shaders");
    Ok(())
}

fn compile_modules(
    compiler: &Compiler,
    source_dir: &Path,
    output_dir: &Path,
    modules: &[(&str, &str)],
    binding_file: &str,
) -> Result<()> {
    let mut bindings = String::new();
    for &(module, source) in modules {
        for (stage, profile) in [("vertex", "vs_4_1"), ("fragment", "ps_4_1")] {
            let entry = format!("{module}_{stage}");
            let output = output_dir.join(format!("{entry}.dxbc"));
            if output.exists() {
                fs::remove_file(&output)?;
            }
            compiler.compile(source_dir, source, &entry, profile, &output)?;
            let bytes = fs::read(&output).map_err(|error| {
                format!(
                    "shader compiler did not produce readable output for {entry} at {}: {error}",
                    output.display()
                )
            })?;
            if bytes.len() < 32 || !bytes.starts_with(b"DXBC") {
                return Err(format!("{entry} did not produce DXBC bytecode").into());
            }
            bindings.push_str(&format!(
                "const {}_BYTES: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{entry}.dxbc\"));\n",
                entry.to_ascii_uppercase(),
            ));
        }
    }
    fs::File::create(output_dir.join(binding_file))?.write_all(bindings.as_bytes())?;
    Ok(())
}

enum Compiler {
    #[cfg(target_os = "windows")]
    Fxc(PathBuf),
    Vkd3d(PathBuf),
}

impl Compiler {
    fn find() -> Result<Self> {
        // Explicit portable compiler selection also works on Windows hosts.
        if let Some(path) = env::var_os("GPUI_VKD3D_COMPILER") {
            return Ok(Self::Vkd3d(path.into()));
        }
        #[cfg(target_os = "windows")]
        {
            Ok(Self::Fxc(find_fxc_compiler()?))
        }
        #[cfg(not(target_os = "windows"))]
        {
            Ok(Self::Vkd3d(PathBuf::from("vkd3d-compiler")))
        }
    }

    fn compile(
        &self,
        source_dir: &Path,
        source: &str,
        entry: &str,
        profile: &str,
        output: &Path,
    ) -> Result<()> {
        let mut command = match self {
            #[cfg(target_os = "windows")]
            Self::Fxc(path) => {
                let mut command = Command::new(path);
                command
                    .args(["/nologo", "/T", profile, "/E", entry, "/O3", "/Fo"])
                    .arg(output)
                    .arg(source);
                command
            }
            Self::Vkd3d(path) => {
                let mut command = Command::new(path);
                command
                    .args([
                        "-x", "hlsl", "-b", "dxbc-tpf", "-p", profile, "-e", entry, "-o",
                    ])
                    .arg(output)
                    .arg(source);
                command
            }
        };
        // vkd3d resolves quoted includes against the working directory.
        let result = command.current_dir(source_dir).output().map_err(|error| {
            format!("failed to run shader compiler for {entry}: {error}. Windows hosts need FXC from the Windows SDK; other hosts need vkd3d-compiler with HLSL support (2.1 or later), optionally set by GPUI_VKD3D_COMPILER")
        })?;
        if !result.status.success() {
            return Err(format!(
                "shader compilation failed for {entry} ({}):\n{}\n{}",
                result.status,
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(target_os = "windows")]
fn find_fxc_compiler() -> Result<PathBuf> {
    if let Some(path) = env::var_os("GPUI_FXC_PATH") {
        return Ok(path.into());
    }
    if let Ok(output) = Command::new("where.exe").arg("fxc.exe").output()
        && output.status.success()
        && let Some(path) = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(PathBuf::from)
            .find(|path| path.is_file())
    {
        return Ok(path);
    }
    let key = windows_registry::LOCAL_MACHINE
        .open("SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0")?;
    let install_folder: String = key.get_string("InstallationFolder")?;
    let bin = Path::new(&install_folder).join("bin");
    let mut versions: Vec<_> = fs::read_dir(&bin)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .collect();
    versions.sort_by_key(|entry| {
        entry
            .file_name()
            .to_string_lossy()
            .split('.')
            .filter_map(|part| part.parse::<u32>().ok())
            .collect::<Vec<_>>()
    });
    let arch = match env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "x86",
        architecture => {
            return Err(format!("unsupported FXC host architecture: {architecture}").into());
        }
    };
    versions
        .into_iter()
        .rev()
        .map(|entry| entry.path().join(arch).join("fxc.exe"))
        .find(|path| path.is_file())
        .ok_or_else(|| "FXC not found; install the Windows SDK or set GPUI_FXC_PATH".into())
}
