(() => {
  const tip = document.getElementById("tooltip");
  const $ = (id) => document.getElementById(id);
  const pct = (v) => `${Math.round(v * 100)}%`;

  function showTip(e, text) {
    tip.textContent = text; tip.style.opacity = 1;
    tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY - 30}px`;
  }
  const hideTip = () => { tip.style.opacity = 0; };

  // Horizontal bars: rows = [{name, value, display, tip, color}]
  function bars(container, rows, max) {
    container.replaceChildren();
    max = max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1);
    for (const r of rows) {
      const row = document.createElement("div"); row.className = "hbar";
      const name = document.createElement("span"); name.className = "name"; name.textContent = r.name; name.title = r.name;
      const track = document.createElement("div"); track.className = "track";
      const fill = document.createElement("div"); fill.className = "fill";
      fill.style.width = `${(Math.abs(r.value) / max) * 100}%`;
      if (r.color) fill.style.background = r.color;
      track.append(fill);
      const val = document.createElement("span"); val.className = "val"; val.textContent = r.display;
      row.append(name, track, val);
      row.addEventListener("mousemove", (e) => showTip(e, r.tip));
      row.addEventListener("mouseleave", hideTip);
      container.append(row);
    }
  }
  const rateRows = (list) => list.map((g) => ({
    name: g.group, value: g.rate, display: `${g.rate}%`, tip: `${g.group}: ${g.rate}% of ${g.count} patients`,
  }));

  fetch("/api/insights").then((r) => r.json()).then((d) => {
    const h = d.heart, m = d.model, t = m.test, kb = d.knowledge_base;
    $("k-total").textContent = h.total;
    $("k-rate").textContent = `${Math.round((h.with_disease / h.total) * 100)}%`;
    $("k-rate-sub").textContent = `${h.with_disease} of ${h.total} patients`;
    $("k-acc").textContent = pct(t.accuracy);
    $("k-acc-sub").textContent = `on ${m.n_test} unseen patients`;
    $("k-auc").textContent = t.roc_auc.toFixed(2);

    bars($("c-age"), rateRows(h.by_age), 100);
    bars($("c-cp"), rateRows(h.by_cp), 100);
    bars($("c-sex"), rateRows([...h.by_sex, ...h.by_exang]), 100);

    const tbody = $("t-compare").querySelector("tbody");
    for (const c of h.compare) {
      const tr = document.createElement("tr");
      const diff = (c.disease - c.healthy).toFixed(1);
      for (const [v, cls] of [[c.metric, ""], [c.healthy, "num"], [c.disease, "num"], [`${diff > 0 ? "+" : ""}${diff}`, "num"]]) {
        const td = document.createElement("td"); td.textContent = v; if (cls) td.className = cls; tr.append(td);
      }
      tbody.append(tr);
    }

    const cm = t.confusion_matrix;
    $("eval-sub").textContent = `Hold-out test set (${m.n_test} patients) · precision ${pct(t.precision)} · recall ${pct(t.recall)} · F1 ${pct(t.f1)}`;
    const cells = [
      ["", "h"], ["Predicted healthy", "h"], ["Predicted disease", "h"],
      ["Actually healthy", "h"], [cm.tn, "cell good", "correct"], [cm.fp, "cell", "false alarm"],
      ["Actually disease", "h"], [cm.fn, "cell", "missed"], [cm.tp, "cell good", "correct"],
    ];
    for (const [v, cls, sub] of cells) {
      const div = document.createElement("div"); div.className = cls;
      if (sub) { const b = document.createElement("b"); b.textContent = v; div.append(b, sub); } else div.textContent = v;
      $("cm").append(div);
    }
    const mt = $("t-models").querySelector("tbody");
    for (const [name, v] of Object.entries(m.comparison)) {
      const tr = document.createElement("tr");
      for (const [val, cls] of [[name + (name === m.model ? " (deployed)" : ""), ""], [v.cv_roc_auc.toFixed(3), "num"], [pct(v.cv_accuracy), "num"]]) {
        const td = document.createElement("td"); td.textContent = val; if (cls) td.className = cls; tr.append(td);
      }
      mt.append(tr);
    }

    bars($("c-coef"), m.coefficients.slice(0, 8).map((c) => ({
      name: c.label, value: c.weight, display: (c.weight > 0 ? "+" : "") + c.weight.toFixed(2),
      tip: `${c.label}: ${c.weight > 0 ? "raises" : "lowers"} risk`,
      color: c.weight > 0 ? "var(--series-2)" : "var(--series-1)",
    })));

    $("kb-sub").textContent = `${kb.conditions} conditions · ${kb.symptoms} symptoms · ${kb.phrases} symptom phrases recognised. Conditions per category:`;
    bars($("c-kb"), kb.by_category.map((c) => ({
      name: c.group, value: c.count, display: String(c.count), tip: `${c.group}: ${c.count} conditions`,
    })));
  });
})();
