use eframe::{App, Frame, egui};
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use ndarray::{Array2, par_azip, s};

const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const STEPS_PER_FRAME: usize = 5;

struct Simulation {
    u: Array2<f32>,
    v: Array2<f32>,
    params: Parameters,
}

#[derive(Clone, Copy, PartialEq)]
struct Parameters {
    f: f32,
    k: f32,
    du: f32,
    dv: f32,
}

impl Default for Parameters {
    fn default() -> Self {
        PRESETS[0].1 // "Mitosis"
    }
}

const PRESETS: [(&str, Parameters); 5] = [
    (
        "Mitosis",
        Parameters {
            f: 0.0545,
            k: 0.062,
            du: 0.2097,
            dv: 0.105,
        },
    ),
    (
        "Coral Growth",
        Parameters {
            f: 0.058,
            k: 0.065,
            du: 0.2097,
            dv: 0.105,
        },
    ),
    (
        "Fingerprints",
        Parameters {
            f: 0.026,
            k: 0.051,
            du: 0.2097,
            dv: 0.105,
        },
    ),
    (
        "Worms and Loops",
        Parameters {
            f: 0.078,
            k: 0.061,
            du: 0.2097,
            dv: 0.105,
        },
    ),
    (
        "Custom",
        Parameters {
            f: 0.03,
            k: 0.06,
            du: 0.2097,
            dv: 0.105,
        },
    ),
];

impl Simulation {
    fn new(params: Parameters) -> Self {
        let mut u = Array2::ones((HEIGHT, WIDTH));
        let mut v = Array2::zeros((HEIGHT, WIDTH));

        let r = (WIDTH as f32 * 0.1) as usize;
        let center = WIDTH / 2;
        u.slice_mut(s![center - r..center + r, center - r..center + r])
            .fill(0.5);
        v.slice_mut(s![center - r..center + r, center - r..center + r])
            .fill(0.25);

        Self { u, v, params }
    }

    fn step(&mut self) {
        let u_lap = self.laplacian(&self.u);
        let v_lap = self.laplacian(&self.v);

        let uvv = &self.u * &self.v * &self.v;

        let du_lap = self.params.du * &u_lap;
        let dv_lap = self.params.dv * &v_lap;

        self.u = &self.u + &du_lap - &uvv + self.params.f * (1.0 - &self.u);
        self.v = &self.v + &dv_lap + &uvv - (self.params.f + self.params.k) * &self.v;
    }

    fn laplacian(&self, grid: &Array2<f32>) -> Array2<f32> {
        let mut lap = Array2::zeros(grid.dim());

        let kernel = [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]];

        par_azip!((index (r, c), cell in &mut lap) {
            let mut sum = 0.0;
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let nr = (r as isize + dr).rem_euclid(HEIGHT as isize) as usize;
                    let nc = (c as isize + dc).rem_euclid(WIDTH as isize) as usize;
                    sum += grid[[nr, nc]] * kernel[(dr + 1) as usize][(dc + 1) as usize];
                }
            }
            *cell = sum;
        });

        lap
    }
}

struct ReactionDiffusionApp {
    simulation: Simulation,
    texture: Option<TextureHandle>,
    current_preset_name: &'static str,
}

impl Default for ReactionDiffusionApp {
    fn default() -> Self {
        let params = Parameters::default();
        let simulation = Simulation::new(params);
        let current_preset_name = PRESETS[0].0;

        Self {
            simulation,
            texture: None,
            current_preset_name,
        }
    }
}

impl App for ReactionDiffusionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        if self.texture.is_none() {
            let image = self.create_image_from_simulation();
            self.texture = Some(ctx.load_texture("sim_texture", image, TextureOptions::NEAREST));
        }

        let mut params_changed = false;

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("🔬 Simulation Parameters");
            ui.separator();

            let mut selected_preset = self.current_preset_name;
            egui::ComboBox::from_label("Load a Preset")
                .selected_text(selected_preset)
                .show_ui(ui, |ui| {
                    for (name, params) in PRESETS.iter() {
                        if ui
                            .selectable_value(&mut selected_preset, *name, *name)
                            .clicked()
                        {
                            self.simulation.params = *params;
                            params_changed = true;
                            self.current_preset_name = name;
                        }
                    }
                });

            ui.separator();

            if ui
                .add(
                    egui::Slider::new(&mut self.simulation.params.f, 0.01..=0.1)
                        .text("Feed Rate (F)"),
                )
                .changed()
            {
                params_changed = true;
                self.current_preset_name = "Custom";
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.simulation.params.k, 0.04..=0.07)
                        .text("Kill Rate (k)"),
                )
                .changed()
            {
                params_changed = true;
                self.current_preset_name = "Custom";
            }

            ui.separator();
            ui.label("Advanced Parameters");
            if ui
                .add(
                    egui::Slider::new(&mut self.simulation.params.du, 0.1..=0.3)
                        .text("Diffusion Rate (U)"),
                )
                .changed()
            {
                params_changed = true;
                self.current_preset_name = "Custom";
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.simulation.params.dv, 0.05..=0.15)
                        .text("Diffusion Rate (V)"),
                )
                .changed()
            {
                params_changed = true;
                self.current_preset_name = "Custom";
            }

            if params_changed {
                self.simulation = Simulation::new(self.simulation.params);
            }
            ui.separator();
            if ui.button("Reset Simulation").clicked() {
                self.simulation = Simulation::new(self.simulation.params);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            for _ in 0..STEPS_PER_FRAME {
                self.simulation.step();
            }

            let image = self.create_image_from_simulation();
            if let Some(texture) = &mut self.texture {
                texture.set(image, TextureOptions::NEAREST);
            }

            let available_size = ui.available_size();
            let aspect_ratio = WIDTH as f32 / HEIGHT as f32;
            let mut display_size = available_size;
            if available_size.x / available_size.y > aspect_ratio {
                display_size.x = available_size.y * aspect_ratio;
            } else {
                display_size.y = available_size.x / aspect_ratio;
            }

            ui.centered_and_justified(|ui| {
                if let Some(texture) = &self.texture {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(display_size));
                }
            });
        });

        ctx.request_repaint();
    }
}

impl ReactionDiffusionApp {
    fn create_image_from_simulation(&self) -> ColorImage {
        let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);
        let v_grid = &self.simulation.v;

        // Simple min/max normalization
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for val in v_grid.iter() {
            if *val < min_val {
                min_val = *val;
            }
            if *val > max_val {
                max_val = *val;
            }
        }
        let range = (max_val - min_val).max(1e-6);

        for r in 0..HEIGHT {
            for c in 0..WIDTH {
                let norm_v = (v_grid[[r, c]] - min_val) / range;
                pixels.push(magma_color(norm_v));
            }
        }
        ColorImage {
            size: [WIDTH, HEIGHT],
            pixels,
        }
    }
}

fn magma_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    // Simplified version of the Magma colormap
    let r = (2.112 * t.powf(0.8) - 2.13 * t.powf(1.6) + 1.018 * t.powf(2.4)).clamp(0.0, 1.0);
    let g = (0.247 * t.powf(0.5) + 1.13 * t.powf(1.0) - 0.379 * t.powf(2.0)).clamp(0.0, 1.0);
    let b = (0.01 + 2.05 * t - 1.05 * t.powf(2.0)).clamp(0.0, 1.0);
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[cfg(target_arch = "wasm32")]
use {wasm_bindgen_futures::JsFuture, web_sys};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Reaction-Diffusion Explorer",
        options,
        Box::new(|_cc| Box::<ReactionDiffusionApp>::default()),
    )
}

#[cfg(target_arch = "wasm32")]
#[allow(non_snake_case)]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let num_threads = web_sys::window()
            .map(|win| win.navigator().hardware_concurrency() as usize)
            .unwrap_or(4); // Default to 4 threads if not available

        log::info!("Using {} threads for wasm.", num_threads);

        // Convert the JS Promise to a Rust Future and await it.
        let promise = wasm_bindgen_rayon::init_thread_pool(num_threads);
        JsFuture::from(promise)
            .await
            .expect("failed to init thread pool");

        eframe::WebRunner::new()
            .start(
                "the-canvas", // hardcode it
                web_options,
                Box::new(|_cc| Box::<ReactionDiffusionApp>::default()),
            )
            .await
            .expect("failed to start eframe");
    });
}
