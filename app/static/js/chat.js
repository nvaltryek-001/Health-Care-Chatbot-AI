/* MedBot chat client. All text is inserted with textContent (no innerHTML from data). */
(() => {
  const $messages = document.getElementById("messages");
  const $form = document.getElementById("composer");
  const $input = document.getElementById("input");
  const $send = document.getElementById("send-btn");
  const $tags = document.getElementById("symptom-tags");
  const STORE_KEY = "medbot-chat-v2";

  let state = { symptoms: [], history: [] };
  let busy = false;

  const store = {
    load() { try { return JSON.parse(sessionStorage.getItem(STORE_KEY)); } catch { return null; } },
    save() { try { sessionStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* ignore */ } },
    clear() { try { sessionStorage.removeItem(STORE_KEY); } catch { /* ignore */ } },
  };

  // ------------------------------------------------------------ DOM helpers
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c != null) node.append(c);
    return node;
  }
  const svg = (markup) => { const t = document.createElement("template"); t.innerHTML = markup.trim(); return t.content.firstChild; };
  const BOT_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="4" y="7" width="16" height="12" rx="3"/><path d="M12 3v4M9 12h.01M15 12h.01"/></svg>';
  const ALERT_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>';
  const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const scroll = () => { $messages.scrollTop = $messages.scrollHeight; };

  // -------------------------------------------------------------- renderers
  function section(title, items, focus = false) {
    if (!items || !items.length) return null;
    return el("div", { class: "section" + (focus ? " focus" : "") },
      el("h4", { text: title }),
      el("ul", {}, items.map((i) => el("li", { text: i }))));
  }

  function urgencyBadge(urgency) {
    const text = { "self-care": "Self-care usually OK", "see-doctor": "See a doctor", urgent: "Urgent" }[urgency] || urgency;
    return el("span", { class: `badge ${urgency}`, text });
  }

  function sectionsFor(c, focus) {
    const map = {
      symptoms: ["Common symptoms", c.symptom_labels],
      treatment: ["What helps", [...c.self_care, ...c.medicines]],
      doctor: [`When to see a doctor · ${c.specialist}`, c.see_doctor],
      prevention: ["Prevention", c.prevention],
    };
    const order = focus && map[focus] ? [focus, ...Object.keys(map).filter((k) => k !== focus)] : Object.keys(map);
    return order.map((k) => section(map[k][0], map[k][1], k === focus));
  }

  function conditionCard(c, rank) {
    const card = el("div", { class: "cond" },
      el("div", { class: "cond-top" },
        el("span", { class: "cond-rank", text: String(rank) }),
        el("h3", { text: c.name }),
        el("span", { class: "badge", text: c.category }),
        urgencyBadge(c.urgency)),
      el("div", { class: "match" },
        el("span", { text: "Symptom match" }),
        el("div", { class: "match-bar" }, el("span", { style: `width:${Math.max(4, c.match)}%` })),
        el("b", { text: `${c.match}%` })),
      el("p", { text: c.summary }),
      c.matched && c.matched.length
        ? el("div", { class: "matched" }, el("b", { text: "Matches: " }), c.matched.join(", "))
        : null);
    const more = el("details", { class: "more" }, el("summary", { text: "Self-care, medicines & when to see a doctor" }),
      el("div", { class: "sections" }, sectionsFor(c, "treatment")));
    card.append(more);
    return card;
  }

  function infoCard(c) {
    return el("div", { class: "cond" },
      el("div", { class: "cond-top" },
        el("h3", { text: c.name }),
        el("span", { class: "badge", text: c.category }),
        urgencyBadge(c.urgency)),
      c.focus !== "general" ? el("p", { text: c.summary }) : null,
      el("div", { class: "sections" }, sectionsFor(c, c.focus === "general" ? "symptoms" : c.focus)));
  }

  function addUser(text, time = now()) {
    $messages.append(el("div", { class: "msg user" },
      el("div", { class: "msg-body" }, el("div", { class: "bubble", text }), el("div", { class: "time", text: time }))));
    scroll();
  }

  function addBot(data, time = now(), interactive = true) {
    const body = el("div", { class: "msg-body" });
    if (data.alert) {
      body.append(el("div", { class: `alert ${data.alert.kind}` }, svg(ALERT_ICON), el("div", { text: data.alert.message })));
    }
    if (!data.alert || data.alert.message !== data.reply) body.append(el("div", { class: "bubble", text: data.reply }));
    if (data.conditions && data.conditions.length) {
      body.append(el("div", { class: "cond-list" }, data.conditions.map((c, i) => conditionCard(c, i + 1))));
    }
    if (data.info) body.append(infoCard(data.info));
    if (data.action) body.append(el("div", {}, el("a", { class: "btn sm", href: data.action.href, text: data.action.label })));
    if (interactive && data.suggestions && data.suggestions.length) {
      body.append(el("div", { class: "chips" }, data.suggestions.map((s) =>
        el("button", { class: "chip", type: "button", text: s.label, onclick: () => send(s.message) }))));
    }
    body.append(el("div", { class: "time", text: time }));
    const node = el("div", { class: "msg bot" }, el("div", { class: "avatar" }, svg(BOT_ICON)), body);
    $messages.append(node);
    // Long answers: show the start of the reply rather than jumping to its end.
    if (node.offsetHeight > $messages.clientHeight * 0.8) $messages.scrollTop = node.offsetTop - $messages.offsetTop - 12;
    else scroll();
  }

  function typing(on) {
    const existing = document.getElementById("typing");
    if (!on) { existing && existing.remove(); return; }
    if (existing) return;
    $messages.append(el("div", { class: "msg bot", id: "typing" },
      el("div", { class: "avatar" }, svg(BOT_ICON)),
      el("div", { class: "bubble typing" }, el("span"), el("span"), el("span"))));
    scroll();
  }

  function renderTags() {
    $tags.replaceChildren();
    if (!state.symptoms.length) {
      $tags.append(el("span", { class: "empty", text: "No symptoms yet. Tell MedBot how you feel." }));
      return;
    }
    for (const s of state.symptoms) {
      $tags.append(el("span", { class: "tag" }, s.label,
        el("button", { type: "button", "aria-label": `Remove ${s.label}`, text: "×",
          onclick: () => removeSymptom(s.id) })));
    }
  }

  // ---------------------------------------------------------------- network
  async function post(payload) {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  async function run(payload, userText) {
    if (busy) return;
    busy = true; $send.disabled = true;
    const time = now();
    if (userText) addUser(userText, time);
    typing(true);
    try {
      const data = await post(payload);
      await new Promise((r) => setTimeout(r, 250));
      typing(false);
      state.symptoms = data.symptoms || [];
      addBot(data);
      if (userText) state.history.push({ role: "user", text: userText, time });
      state.history.push({ role: "bot", data, time: now() });
      state.history = state.history.slice(-60);
      renderTags(); store.save();
    } catch (err) {
      typing(false);
      addBot({ reply: "Sorry, I couldn't reach the server. Please check your connection and try again." });
    } finally {
      busy = false; $send.disabled = false; $input.focus();
    }
  }

  function send(text) {
    text = (text || "").trim();
    if (!text) return;
    run({ message: text, symptoms: state.symptoms.map((s) => s.id) }, text);
  }

  function removeSymptom(id) {
    const remaining = state.symptoms.filter((s) => s.id !== id).map((s) => s.id);
    run({ action: "rerank", message: "", symptoms: remaining }, null);
  }

  // ------------------------------------------------------------------- init
  $form.addEventListener("submit", (e) => { e.preventDefault(); const t = $input.value; $input.value = ""; send(t); });
  document.getElementById("reset-btn").addEventListener("click", () => {
    state = { symptoms: [], history: [] }; store.clear(); $messages.replaceChildren(); renderTags(); greet();
  });

  function greet() {
    addBot({
      reply: "Hi! I'm MedBot, your AI health assistant. 👋\nTell me what symptoms you have and I'll suggest possible conditions, self-care tips and when to see a doctor. You can also ask about a disease, e.g. “how to prevent dengue?”.",
      suggestions: [
        { label: "I have fever and headache", message: "I have fever and headache" },
        { label: "Sore throat and runny nose", message: "sore throat and runny nose" },
        { label: "How to prevent dengue?", message: "How to prevent dengue?" },
        { label: "Check my heart risk", message: "Check my heart risk" },
      ],
    });
  }

  const saved = store.load();
  if (saved && saved.history && saved.history.length) {
    state = saved;
    greet();
    const last = state.history.length - 1;
    state.history.forEach((h, i) => {
      if (h.role === "user") addUser(h.text, h.time);
      if (h.role === "bot") addBot(h.data, h.time, i === last);
    });
    renderTags();
  } else {
    greet();
  }
  $input.focus();
})();
