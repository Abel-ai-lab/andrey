// The navbar's Liquid Glass, in Chromium only: Safari and Firefox accept a url() backdrop-filter
// and paint nothing, so they keep the frosted pill. The lens map is drawn once for the pill's size,
// and again only when that size changes.
//
// The map shifts the backdrop toward the pill's middle in a band along its rounded edge, most at
// the edge and easing to none inside, as a thick pane's rim bends light. Red carries the sideways
// shift and green the vertical one; mid-gray is no shift.
(() => {
  const root = document.documentElement;
  const pill = document.querySelector(".bd-header__inner");
  const map = document.getElementById("andrey-lens-map");
  const chromium = navigator.userAgentData?.brands?.some((b) => b.brand === "Chromium");
  if (!pill || !map || !chromium || matchMedia("(prefers-reduced-transparency: reduce)").matches) return;

  const canvas = document.createElement("canvas");
  let drawn = "";
  const draw = () => {
    const width = Math.round(pill.offsetWidth);
    const height = Math.round(pill.offsetHeight);
    if (!width || !height || drawn === `${width}x${height}`) return;
    drawn = `${width}x${height}`;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    const image = context.createImageData(width, height);
    const radius = height / 2;
    const band = Math.min(radius, 16);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // The nearest point on the pill's spine, a line between the centers of its two ends.
        const cx = Math.min(Math.max(x + 0.5, radius), width - radius);
        const dx = x + 0.5 - cx;
        const dy = y + 0.5 - radius;
        const from = Math.hypot(dx, dy);
        const depth = radius - from;
        let r = 128;
        let g = 128;
        if (depth >= 0 && depth < band && from > 0) {
          const pull = (1 - depth / band) ** 2;
          r = 128 - (127 * pull * dx) / from;
          g = 128 - (127 * pull * dy) / from;
        }
        const i = (y * width + x) * 4;
        image.data[i] = r;
        image.data[i + 1] = g;
        image.data[i + 2] = 128;
        image.data[i + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    map.setAttribute("href", canvas.toDataURL());
    map.setAttribute("width", width);
    map.setAttribute("height", height);
    root.dataset.glass = "liquid";
  };
  draw();
  new ResizeObserver(draw).observe(pill);
})();
