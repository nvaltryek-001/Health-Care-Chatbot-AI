(() => {
  const form = document.getElementById("heart-form");
  const btn = document.getElementById("predict-btn");
  const PRESETS = {
    low: { age: 38, sex: 0, cp: 2, trestbps: 118, chol: 195, fbs: 0, restecg: 1, thalach: 176, exang: 0, oldpeak: 0, slope: 2, ca: 0, thal: 2 },
    high: { age: 64, sex: 1, cp: 0, trestbps: 158, chol: 286, fbs: 1, restecg: 0, thalach: 112, exang: 1, oldpeak: 2.4, slope: 1, ca: 2, thal: 3 },
  };
  const COLORS = { Low: "#2f855a", Moderate: "#b7791f", High: "#c53030" };
  const TEXT = {
    Low: "Your profile looks similar to patients without heart disease in the dataset. Keep up healthy habits.",
    Moderate: "Some risk factors are present. Consider a check-up with your doctor.",
    High: "Your profile resembles patients diagnosed with heart disease. Please consult a cardiologist.",
  };

  function values() {
    const out = {};
    for (const elm of form.elements) if (elm.name) out[elm.name] = Number(elm.value);
    return out;
  }

  function showErrors(fields) {
    form.querySelectorAll(".field").forEach((f) => { f.classList.remove("invalid"); f.querySelector(".err")?.remove(); });
    for (const [name, msg] of Object.entries(fields || {})) {
      const f = form.querySelector(`[data-field="${name}"]`);
      if (!f) continue;
      f.classList.add("invalid");
      const e = document.createElement("div"); e.className = "err"; e.textContent = `Value ${msg}.`; f.append(e);
    }
  }

  function render(r) {
    document.getElementById("result-empty").hidden = true;
    document.getElementById("result-body").hidden = false;
    const arc = document.getElementById("gauge-arc");
    arc.setAttribute("stroke", COLORS[r.level]);
    arc.setAttribute("stroke-dasharray", `${Math.max(1, r.percent)} 100`);
    document.getElementById("risk-pct").textContent = `${r.percent}%`;
    const lvl = document.getElementById("risk-lvl");
    lvl.className = `lvl ${r.level}`; lvl.textContent = `${r.level} risk`;
    document.getElementById("risk-text").textContent = TEXT[r.level];

    const list = document.getElementById("factors");
    list.replaceChildren();
    const max = Math.max(...r.factors.map((f) => Math.abs(f.impact)), 1);
    for (const f of r.factors) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${f.direction === "increases" ? "▲" : "▼"} ${f.label}`;
      name.title = `${f.direction} risk (log-odds ${f.impact > 0 ? "+" : ""}${f.impact})`;
      const bar = document.createElement("div"); bar.className = "fbar";
      const fill = document.createElement("span");
      fill.className = f.impact > 0 ? "up" : "down";
      fill.style.width = `${(Math.abs(f.impact) / max) * 50}%`;
      bar.append(fill); li.append(name, bar); list.append(li);
    }
    if (window.innerWidth < 960) document.getElementById("result").scrollIntoView({ behavior: "smooth" });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true; btn.textContent = "Calculating…";
    try {
      const res = await fetch("/api/heart/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values()),
      });
      const data = await res.json();
      if (!res.ok) { showErrors(data.fields); return; }
      showErrors({}); render(data);
    } catch {
      showErrors({});
      document.getElementById("result-empty").querySelector("p").textContent = "Could not reach the server. Please try again.";
    } finally {
      btn.disabled = false; btn.textContent = "Estimate my risk";
    }
  });

  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const p = PRESETS[b.dataset.preset];
    for (const [k, v] of Object.entries(p)) if (form.elements[k]) form.elements[k].value = v;
    form.requestSubmit();
  }));
})();
