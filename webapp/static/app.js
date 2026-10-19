(function () {
  "use strict";

  // ---------- theme (remembered if storage is available)
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) { /* storage unavailable: ignore */ }
  var themeBtn = document.getElementById("theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      var current = root.getAttribute("data-theme") || (dark ? "dark" : "light");
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    });
  }

  var drop = document.getElementById("drop");
  if (!drop) return; // model not loaded: page only shows the warning

  var fileInput = document.getElementById("file");
  var preview = document.getElementById("preview");
  var errorEl = document.getElementById("error");
  var result = document.getElementById("result");

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }

  drop.addEventListener("click", function () { fileInput.click(); });
  drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) handle(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) handle(fileInput.files[0]);
  });

  function handle(file) {
    showError("");
    if (!/^image\//.test(file.type)) { showError("Please choose an image file."); return; }
    if (file.size > 8 * 1024 * 1024) { showError("Image is too large (limit 8 MB)."); return; }

    var url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    drop.classList.add("has-img", "busy");

    var form = new FormData();
    form.append("file", file);
    fetch("/api/predict", { method: "POST", body: form })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error(res.body.error || "Prediction failed.");
        render(res.body);
      })
      .catch(function (err) { showError(err.message || "Something went wrong."); result.hidden = true; })
      .finally(function () { drop.classList.remove("busy"); fileInput.value = ""; });
  }

  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function render(r) {
    var confident = r.is_confident;
    var notFood = r.is_food === false;
    document.getElementById("eyebrow").textContent = notFood ? "Not food" : (confident ? "Prediction" : "Best guess (not sure)");
    document.getElementById("label").textContent = notFood ? "This doesn't look like food" : (confident ? r.display_label : "Not sure");
    document.getElementById("confval").textContent = pct(notFood ? 1 - r.food_score : r.confidence);
    document.querySelector(".conf small").textContent = notFood ? "not food" : "confidence";
    var unsure = document.getElementById("unsure");
    unsure.hidden = confident;
    unsure.textContent = notFood
      ? "This image doesn't look like a photo of food, so no dish is suggested. Try a clear photo of a dish."
      : "The model isn't confident enough to name this dish. It may not be one of the supported foods, or the photo may be unclear. The closest guesses are below.";
    document.querySelector("#result .grid").hidden = notFood;

    var bars = document.getElementById("bars");
    bars.textContent = "";
    r.topk.forEach(function (t) {
      var li = document.createElement("li");
      var row = document.createElement("div"); row.className = "row";
      var a = document.createElement("span"); a.textContent = t.display_label;
      var b = document.createElement("span"); b.textContent = pct(t.confidence);
      row.appendChild(a); row.appendChild(b);
      var track = document.createElement("div"); track.className = "track";
      var fill = document.createElement("div"); fill.className = "fill"; fill.style.width = "0%";
      track.appendChild(fill);
      li.appendChild(row); li.appendChild(track);
      bars.appendChild(li);
      requestAnimationFrame(function () { fill.style.width = (t.confidence * 100).toFixed(1) + "%"; });
    });

    var nut = document.getElementById("nutrition");
    if (r.nutrition && confident) {
      nut.hidden = false;
      nut.textContent = "";
      var strong = document.createElement("b");
      strong.textContent = r.nutrition.kcal_min + "–" + r.nutrition.kcal_max + " kcal";
      nut.appendChild(strong);
      nut.appendChild(document.createTextNode(" per " + r.nutrition.serving + " (" + r.nutrition.note + ")"));
    } else {
      nut.hidden = true;
    }

    document.getElementById("view").src = "data:image/png;base64," + r.model_view_png_b64;
    document.getElementById("cam").src = "data:image/png;base64," + r.gradcam_png_b64;
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------- model card
  fetch("/api/model-info").then(function (r) { return r.ok ? r.json() : null; }).then(function (m) {
    if (!m) return;
    var tm = m.test_metrics || {};
    var items = [
      ["Architecture", m.arch],
      ["Parameters", (m.params / 1e6).toFixed(2) + " M"],
      ["Input size", m.img_size + " × " + m.img_size],
      ["Test accuracy", tm.accuracy != null ? pct(tm.accuracy) : "n/a"],
      ["Top-3 accuracy", tm.top3_accuracy != null ? pct(tm.top3_accuracy) : "n/a"],
      ["Macro F1", tm.macro_f1 != null ? tm.macro_f1.toFixed(3) : "n/a"],
      ["Reject below", pct(m.reject_threshold)],
    ];
    var dl = document.getElementById("mc");
    items.forEach(function (it) {
      var d = document.createElement("div");
      var dt = document.createElement("dt"); dt.textContent = it[0];
      var dd = document.createElement("dd"); dd.textContent = it[1];
      d.appendChild(dt); d.appendChild(dd); dl.appendChild(d);
    });
    document.getElementById("modelcard").hidden = false;
  }).catch(function () { /* optional */ });
})();
