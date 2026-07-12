(function () {
  const langSelect = document.getElementById("langSelect");
  const sourceInput = document.getElementById("sourceInput");
  const refInput = document.getElementById("refInput");
  const btn = document.getElementById("translateBtn");
  const status = document.getElementById("translateStatus");
  const samplesList = document.getElementById("samplesList");
  const pairTitle = document.getElementById("pairTitle");
  const sourceLabel = document.getElementById("sourceLabel");

  const outInput = document.getElementById("outInput");
  const outTranslation = document.getElementById("outTranslation");
  const outBleu = document.getElementById("outBleu");

  let modelReady = false;
  let currentPair = langSelect.value;

  function setStatus(text, kind) {
    status.textContent = text;
    status.className = "status-line" + (kind ? ` is-${kind}` : "");
  }

  function updatePairUI() {
    const [src, trg] = currentPair.split("-");
    sourceLabel.textContent = `${src.toUpperCase()} sentence`;
    sourceInput.placeholder = `e.g. ${currentPair === "de-en" ? "der mann geht" : currentPair === "en-de" ? "the man is walking" : currentPair === "fr-en" ? "l'homme marche" : currentPair === "en-fr" ? "the man is walking" : currentPair === "es-en" ? "el hombre camina" : currentPair === "en-es" ? "the man is walking" : currentPair === "it-en" ? "l'uomo cammina" : "the man is walking"}`;
    pairTitle.innerHTML = `${currentPair === "de-en" ? "German" : currentPair === "en-de" ? "English" : currentPair === "fr-en" ? "French" : currentPair === "en-fr" ? "English" : currentPair === "es-en" ? "Spanish" : currentPair === "en-es" ? "English" : currentPair === "it-en" ? "Italian" : "English"} <span class="accent">&rarr;</span> ${currentPair.endsWith("en") ? "English" : currentPair.endsWith("de") ? "German" : currentPair.endsWith("fr") ? "French" : currentPair.endsWith("es") ? "Spanish" : currentPair.endsWith("it") ? "Italian" : "English"}`;
  }

  async function ensureModel() {
    if (modelReady && currentPair === langSelect.value) return;
    currentPair = langSelect.value;
    updatePairUI();
    setStatus("Training the Seq2Seq translator for the selected language pair…");
    btn.disabled = true;
    const data = await postJSON("/api/train-translator", { lang_pair: currentPair });
    modelReady = true;
    btn.disabled = false;
    setStatus(
      `Model ready — ${data.src_vocab_size} ${currentPair.split("-")[0].toUpperCase()} tokens, ${data.trg_vocab_size} ${currentPair.split("-")[1].toUpperCase()} tokens.`,
      "ready"
    );
    renderSamples(data.examples);
  }

  function renderSamples(examples) {
    samplesList.innerHTML = "";
    examples.forEach((ex) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "sample-chip";
      chip.textContent = ex.src;
      chip.addEventListener("click", () => {
        sourceInput.value = ex.src;
        refInput.value = ex.trg;
        translate();
      });
      samplesList.appendChild(chip);
    });
  }

  async function translate() {
    const text = sourceInput.value.trim();
    const reference = refInput.value.trim();
    if (!text) return;

    try {
      await ensureModel();
      setStatus("Translating…");
      const data = await postJSON("/api/translate", { text, reference, lang_pair: currentPair });

      outInput.textContent = data.input;
      outTranslation.textContent = data.translation || "(empty output)";
      outBleu.textContent = "bleu" in data ? data.bleu.toFixed(4) : "— (add a reference to score)";

      setStatus("Translation complete.", "ready");
    } catch (e) {
      setStatus(e.message, "error");
    }
  }

  btn.addEventListener("click", translate);
  sourceInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) translate();
  });
  langSelect.addEventListener("change", () => {
    currentPair = langSelect.value;
    updatePairUI();
    modelReady = false;
    samplesList.innerHTML = "";
  });

  updatePairUI();

  // Pre-warm the sample chip list without blocking the page.
  getJSON("/api/vocab").catch(() => {});
})();
