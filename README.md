# PixelForge

Estúdio local de geração de sprites em pixel art. A interface, a API e a geração rodam nesta máquina; nenhuma chamada à API do PixelLab ou a um serviço de geração remoto é feita.

O motor é o [Pixie P32](https://huggingface.co/EmilioSchwaiger/pixie-32), que gera sprites nativos de 32×32 pixels. O modelo e o código são Apache 2.0. O encoder de texto CLIP é distribuído separadamente sob MIT. Os arquivos de licença e atribuição ficam junto aos pesos em `models/` e ao código em `vendor/pixie/`.

O fluxo de assets agora inclui **Bíblias visuais** reutilizáveis, catálogo local de sprites aprovados e exportação para Godot. Os registros são guardados em SQLite (`outputs/pixelforge.db`); as imagens de origem continuam em `outputs/generated`.

## Hardware e limites

- Configurado para a GTX 1060 6 GB desta máquina, com PyTorch CUDA 12.6.
- Descrições curtas em inglês dão melhores resultados.
- Saída nativa é 32×32; o PNG é ampliado com vizinho mais próximo para 256×256 ou 512×512.
- É um modelo pequeno de pesquisa: a qualidade varia e podem aparecer pixels isolados. Não gera animações, tilesets ou personagens em múltiplas direções nesta versão.

## Rodar

Neste workspace, o ambiente `.venv` e os pesos do modelo já estão instalados. Execute `./start.ps1` para iniciar a API em segundo plano e abrir a interface em `http://127.0.0.1:8765`.

Para preparar outra cópia do projeto, instale Python 3.13 e execute `./setup.ps1`. O script instala PyTorch CUDA 12.6 para GPUs GTX 10, instala as dependências e baixa os pesos para `models/`. A primeira preparação baixa cerca de 5 GB entre runtime e modelos. Depois, `./start.ps1` abre a interface. Os PNGs gerados ficam em `outputs/generated`.

## API

- `GET /api/health` — mostra se os pesos locais estão presentes e qual dispositivo será usado.
- `POST /api/generate` — recebe `{ "prompt": "red mushroom with golden spots", "scale": 16, "seed": 42 }` (`seed` é opcional) e devolve duas opções com seeds reproduzíveis. Repetir prompt, seed, escala e revisão do modelo serve o par do cache local (`outputs/cache`) sem rodar o modelo de novo.
- `GET /api/images/{id}` — entrega um PNG gerado.

O endpoint aceita prompts de 2 a 500 caracteres e escalas de 4× a 32×; a interface oferece 8× e 16×. Para solicitar novas opções, limpe a seed no campo da interface. Seeds são derivadas da seed base e cada sprite pode ser baixado separadamente. A interface permite salvar estilos compostos por nome e regras visuais, aplicar um estilo durante a geração e promover a opção escolhida ao catálogo. **Exportar para Godot** copia os assets aprovados para `outputs/godot-assets` e cria `pixelforge-assets.json` com prompt, seed e estilo para rastreabilidade.

Endpoints adicionais: `GET/POST /api/styles`, `GET/POST /api/assets` (POST promove uma imagem) e `POST /api/export/godot`. A sincronização Git, worker em fila e gestão de conflitos descritos no blueprint do GameForge não fazem parte desta etapa.
