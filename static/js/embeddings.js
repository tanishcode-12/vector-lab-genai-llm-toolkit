(function () {
  const input = document.getElementById("wordInput");
  const btn = document.getElementById("queryBtn");
  const status = document.getElementById("status");
  const vocabHint = document.getElementById("vocabHint");
  const neighboursList = document.getElementById("neighboursList");
  const plotWrap = document.getElementById("plotWrap");

  let modelReady = false;

  function setStatus(text, kind) {
    status.textContent = text;
    status.className = "status-line" + (kind ? ` is-${kind}` : "");
  }

  async function ensureModel() {
    if (modelReady) return;
    setStatus("Training CBOW on the toy corpus… this happens once per session.");
    btn.disabled = true;
    const data = await postJSON("/api/train-embeddings", {});
    modelReady = true;
    btn.disabled = false;
    setStatus(`Model ready — vocabulary size ${data.vocab_size}.`, "ready");
    loadVocabHint();
  }

  async function loadVocabHint() {
    try {
      const data = await getJSON("/api/vocab");
      const sample = data.words.slice(0, 24);
      vocabHint.innerHTML =
        "Try: " +
        sample
          .map((w) => `<span data-word="${w}">${w}</span>`)
          .join("");
      vocabHint.querySelectorAll("span").forEach((el) => {
        el.addEventListener("click", () => {
          input.value = el.dataset.word;
          runQuery();
        });
      });
    } catch (e) {
      /* non-critical */
    }
  }

  function renderNeighbours(neighbours) {
    neighboursList.innerHTML = "";
    if (!neighbours.length) {
      neighboursList.innerHTML = `<li>No neighbours found.</li>`;
      return;
    }
    const maxScore = Math.max(...neighbours.map((n) => n.similarity), 0.001);
    neighbours.forEach((n) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <div style="flex:1">
          <div style="display:flex; justify-content:space-between;">
            <span class="neighbour-word">${n.word}</span>
            <span class="neighbour-score">${n.similarity.toFixed(3)}</span>
          </div>
          <div class="neighbour-bar" style="width:${(n.similarity / maxScore) * 100}%"></div>
        </div>`;
      neighboursList.appendChild(li);
    });
  }

  async function loadPlot() {
    plotWrap.innerHTML = `<p class="plot-placeholder">Rendering t-SNE projection…</p>`;
    try {
      const data = await getJSON("/api/embedding-plot");
      plotWrap.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="t-SNE projection of word embeddings">`;
    } catch (e) {
      plotWrap.innerHTML = `<p class="plot-placeholder">Could not render plot: ${e.message}</p>`;
    }
  }

  async function runQuery() {
    const word = input.value.trim().toLowerCase();
    if (!word) return;

    try {
      await ensureModel();
      setStatus(`Searching neighbours for "${word}"…`);
      const data = await postJSON("/api/similar-words", { word });

      if (data.error) {
        setStatus(data.error, "error");
        neighboursList.innerHTML = "";
        return;
      }

      renderNeighbours(data.neighbours);
      setStatus(`Showing ${data.neighbours.length} nearest neighbours for "${word}".`, "ready");
      loadPlot();
    } catch (e) {
      setStatus(e.message, "error");
    }
  }

  btn.addEventListener("click", runQuery);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runQuery();
  });
})();
