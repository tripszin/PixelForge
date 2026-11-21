const form = document.querySelector('#generator');
const button = document.querySelector('#submit');
const candidateArea = document.querySelector('#candidate-area');
const download = document.querySelector('#download');
const newAttempt = document.querySelector('#new-attempt');
const errorBox = document.querySelector('#error');
const engine = document.querySelector('.engine');
const engineStatus = document.querySelector('#engine-status');
const seedInput = document.querySelector('#seed-input');
const seedLabel = document.querySelector('#seed');
const styleSelect = document.querySelector('#style-select');
const promoteButton = document.querySelector('#promote');
const styleForm = document.querySelector('#style-form');
const libraryMessage = document.querySelector('#library-message');
let selectedCandidate = null;

async function loadLibrary() {
  const [stylesResponse, assetsResponse] = await Promise.all([fetch('/api/styles'), fetch('/api/assets')]);
  if (!stylesResponse.ok || !assetsResponse.ok) return;
  const styles = await stylesResponse.json();
  const assets = await assetsResponse.json();
  const selectedStyle = styleSelect.value;
  styleSelect.replaceChildren(new Option('Sem estilo salvo', ''));
  styles.forEach((style) => styleSelect.add(new Option(style.name, style.id)));
  if (styles.some((style) => style.id === selectedStyle)) styleSelect.value = selectedStyle;
  document.querySelector('#asset-count').textContent = assets.length ? `${assets.length} sprite${assets.length === 1 ? '' : 's'} no catálogo.` : 'Nenhum sprite promovido ainda.';
  const list = document.querySelector('#asset-list');
  list.replaceChildren();
  assets.forEach((asset) => {
    const item = document.createElement('a');
    item.className = 'asset-item';
    item.href = `/api/images/${asset.image_id}`;
    item.target = '_blank';
    item.rel = 'noreferrer';
    const image = document.createElement('img');
    image.src = item.href;
    image.alt = asset.prompt;
    const description = document.createElement('span');
    description.textContent = `${asset.prompt}${asset.style_name ? ` · ${asset.style_name}` : ''}`;
    item.append(image, description);
    list.append(item);
  });
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    engine.classList.toggle('online', data.engine === 'ready');
    engine.classList.toggle('offline', data.engine !== 'ready');
    engineStatus.textContent = data.engine === 'ready' ? `modelo local · ${data.device}` : 'modelo ausente';
  } catch {
    engine.classList.add('offline');
    engineStatus.textContent = 'api desconectada';
  }
}

function selectCandidate(candidate, index, cards) {
  selectedCandidate = candidate;
  cards.forEach((card, cardIndex) => {
    const selected = cardIndex === index;
    card.classList.toggle('selected', selected);
    card.setAttribute('aria-pressed', String(selected));
  });
  download.href = candidate.url;
  download.download = `pixelforge-seed-${candidate.seed}.png`;
  download.hidden = false;
  promoteButton.hidden = false;
  seedLabel.textContent = `SEED ${candidate.seed}`;
}

function showCandidates(data) {
  candidateArea.replaceChildren();
  candidateArea.classList.remove('loading');
  const cards = data.candidates.map((candidate, index) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'candidate';
    card.setAttribute('aria-pressed', 'false');
    const img = document.createElement('img');
    img.src = candidate.url;
    img.alt = `Sprite da opção ${index + 1}`;
    const info = document.createElement('span');
    info.className = 'candidate-info';
    const title = document.createElement('strong');
    title.textContent = `OPÇÃO ${index + 1}`;
    const seed = document.createElement('small');
    seed.textContent = `SEED ${candidate.seed}`;
    info.append(title, seed);
    card.append(img, info);
    card.addEventListener('click', () => selectCandidate(candidate, index, cards));
    return card;
  });
  candidateArea.append(...cards);
  selectCandidate(data.candidates[0], 0, cards);
  seedInput.value = String(data.base_seed);
  seedLabel.textContent = `BASE ${data.base_seed}${data.cache_hit ? ' · CACHE' : ''}`;
  newAttempt.hidden = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const seedText = seedInput.value.trim();
  const payload = {
    prompt: document.querySelector('#prompt').value.trim(),
    scale: Number(document.querySelector('#size').value),
    style_id: styleSelect.value || null,
  };
  if (seedText !== '') payload.seed = Number(seedText);

  errorBox.hidden = true;
  download.hidden = true;
  newAttempt.hidden = true;
  promoteButton.hidden = true;
  seedLabel.textContent = '';
  candidateArea.classList.add('loading');
  candidateArea.innerHTML = '<div class="placeholder"><div class="pixel-icon">✳</div><p>Gerando duas opções…</p><span>A primeira geração carrega o modelo na memória</span></div>';
  button.disabled = true;
  button.firstElementChild.textContent = 'Gerando…';
  try {
    const response = await fetch('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Não foi possível gerar as opções.');
    showCandidates(data);
  } catch (error) {
    candidateArea.classList.remove('loading');
    candidateArea.innerHTML = '<div class="placeholder"><div class="pixel-icon">!</div><p>Não foi possível gerar agora</p><span>Confira os arquivos do modelo local</span></div>';
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = 'Gerar duas opções';
  }
});

styleForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  libraryMessage.textContent = '';
  const response = await fetch('/api/styles', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: document.querySelector('#style-name').value.trim(), bible: document.querySelector('#style-bible').value.trim()})});
  const data = await response.json();
  if (!response.ok) { libraryMessage.textContent = data.detail || 'Não foi possível salvar o estilo.'; return; }
  styleForm.reset();
  styleSelect.value = data.id;
  libraryMessage.textContent = 'Estilo visual salvo.';
  await loadLibrary();
  styleSelect.value = data.id;
});

promoteButton.addEventListener('click', async () => {
  if (!selectedCandidate) return;
  const response = await fetch('/api/assets', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({image_id: selectedCandidate.id, seed: selectedCandidate.seed, prompt: document.querySelector('#prompt').value.trim(), style_id: styleSelect.value || null})});
  const data = await response.json();
  libraryMessage.textContent = response.ok ? 'Sprite promovido para o catálogo do projeto.' : (data.detail || 'Não foi possível promover o sprite.');
  if (response.ok) await loadLibrary();
});

document.querySelector('#export-assets').addEventListener('click', async () => {
  const response = await fetch('/api/export/godot', {method: 'POST'});
  const data = await response.json();
  libraryMessage.textContent = response.ok ? `${data.count} asset(s) exportados para ${data.path}` : (data.detail || 'Falha ao exportar.');
});

newAttempt.addEventListener('click', () => {
  seedInput.value = '';
  form.requestSubmit();
});

checkHealth();
loadLibrary();
setInterval(checkHealth, 30000);
