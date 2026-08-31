use yew::Renderer;

fn main() {
    // Yew 0.23 installs a sensible panic hook by default.
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("no document");
    let root = doc
        .get_element_by_id("app")
        .expect("#app element not found");
    let root: web_sys::Element = root.into();

    Renderer::<hl2_ui::App>::with_root(root).render();
}
