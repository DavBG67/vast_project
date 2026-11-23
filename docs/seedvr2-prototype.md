# Phase 6A.2 - Prototype Docker autonome SeedVR2 (expérimental)

## Objectif

Valider la **faisabilité technique** de SeedVR2 3B FP8 sur GPU 16 Go (RTX 5080 visée) sans modifier le pipeline stable Real-ESRGAN.

Ce prototype est isolé:

- Dockerfile dédié: `prototype/seedvr2/Dockerfile`
- Runner dédié: `scripts/seedvr2_prototype.py`
- Module dédié: `src/vast_upscaler/seedvr2_prototype.py`
- Aucune modification de l'orchestrateur/FastAPI/worker stable.

## Ce que fait le runner prototype

Pour chaque exécution, il:

1. extrait un segment local (`ffmpeg`) ;
2. lance l'inférence SeedVR2 CLI (`/opt/seedvr2/inference_cli.py`) avec réglages low‑VRAM:
   - `attention_mode=sdpa`
   - offload CPU (DiT + VAE + tenseurs)
   - BlockSwap (`blocks_to_swap=32` + `swap_io_components`)
   - VAE tiled encode/decode
   - chunking temporel ;
3. remuxe en sortie finale **MP4 H.264/AAC** (`ffmpeg`) ;
4. produit:
   - log texte exploitable (`run.log`)
   - rapport JSON structuré (`prototype-report.json`) avec paramètres/checkpoints/commandes/stades/erreurs/OOM.

## Checkpoints utilisés

- DiT par défaut: `seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- VAE attendu: `ema_vae_fp16.safetensors`

Identité vérifiée contre le commit SeedVR2 épinglé:

- `seedvr2_ema_3b_fp8_e4m3fn.safetensors`
  - repo: `numz/SeedVR2_comfyUI`
  - SHA256: `3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff`
  - URL: `https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `ema_vae_fp16.safetensors`
  - repo: `numz/SeedVR2_comfyUI`
  - SHA256: `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`
  - URL: `https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors`

Le commit de code SeedVR2 utilisé dans l'image est figé via:

- `ARG NUMZ_REPO_REF=4490bd1f482e026674543386bb2a4d176da245b9`

## Versions verrouillées (prototype)

- Base image: `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime`
- Vérification build:
  - `torch==2.14.1`
  - `torchvision==0.29.1`
  - `torch.version.cuda==13.0`
- Contraintes pip figées:
  - `prototype/seedvr2/requirements.constraints.txt`
- Installation Python isolée:
  - venv dédié: `/opt/seedvr2-venv`
  - `VENV_PATH` + `PATH` pointent vers le venv
  - `python` à l'exécution résout vers `/opt/seedvr2-venv/bin/python`
  - les dépendances `torch`/`torchvision` du `requirements.txt` amont sont filtrées à l'installation pour éviter toute réinstallation involontaire.
- Vérification build additionnelle:
  - `python /opt/seedvr2/inference_cli.py --help` (depuis le venv)
  - présence des options critiques (`--dit_model`, `--attention_mode`, `--blocks_to_swap`, `--vae_encode_tiled`).

## Préflight GPU au démarrage

Avant extraction/inférence, le runner exécute un préflight CUDA:

- journalise `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_arch_list()`, nom GPU et capacité détectée;
- échoue explicitement si CUDA est indisponible;
- échoue explicitement si `sm_120` n'est pas présent dans la liste d'architectures compilées PyTorch.

## Build local (validation statique)

```powershell
docker build -f prototype/seedvr2/Dockerfile -t vast-seedvr2-proto:exp-6a2-v1 .
```

## Build CI GitHub Actions (sans push GHCR)

Workflow dédié: `.github/workflows/seedvr2-docker.yml`

- déclenchement:
  - automatique sur push des branches `seedvr2/**` (avec filtres de chemins SeedVR2) ;
  - manuel via `workflow_dispatch` conservé ;
- construit exclusivement `prototype/seedvr2/Dockerfile`;
- n'effectue aucun push GHCR;
- n'exécute aucune inférence GPU;
- vérifie après build:
  - présence de l'image locale;
  - démarrage de la CLI prototype via `--help`;
  - exécutable Python du venv + versions `torch` / `torchvision` / CUDA compilée.

Déclenchement depuis GitHub:

1. Ouvrir **Actions**.
2. Sélectionner **seedvr2-docker**.
3. Cliquer **Run workflow**.
4. Laisser les valeurs par défaut:
   - `dockerfile=prototype/seedvr2/Dockerfile`
   - `image_name=vast-seedvr2-proto:ci-smoke`
5. Lancer.

Le workflow échoue explicitement si:

- le chemin Dockerfile n'est pas celui du prototype;
- le build Docker échoue;
- l'image construite est introuvable après build.

## Publication GHCR expérimentale (manuelle uniquement)

Workflow dédié: `.github/workflows/seedvr2-ghcr-publish.yml`

- déclenchement: `workflow_dispatch` uniquement;
- Dockerfile imposé: `prototype/seedvr2/Dockerfile`;
- dépôt d'image imposé: `ghcr.io/davbg67/vast-video-upscaler-seedvr2`;
- tag immuable: `exp-6a2-<SHA complet du commit>`;
- publication sous `latest` interdite;
- publication vers l'image stable Real-ESRGAN interdite;
- authentification GHCR via `${{ secrets.GITHUB_TOKEN }}`;
- permissions minimales: `contents: read`, `packages: write`;
- build + smoke CLI (`--help`) validés avant push;
- vérification post-push de la présence distante de l'image + affichage du tag exact dans le résumé.

### Procédure de publication

1. Ouvrir **Actions**.
2. Sélectionner **seedvr2-ghcr-publish**.
3. Cliquer **Run workflow** (aucun paramètre requis).
4. Vérifier dans le résumé:
   - la référence publiée complète (`ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-...`);
   - l'absence de publication `latest`;
   - l'absence de publication vers le dépôt stable Real-ESRGAN.

### Procédure de récupération de l'image publiée

Récupérer la référence exacte affichée dans le résumé du workflow, puis:

```bash
docker pull ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-<sha-commit>
docker run --rm ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-<sha-commit> --help
```

## Cache persistant des poids

Utiliser un volume hôte monté dans `/workspace/cache/seedvr2`:

```powershell
New-Item -ItemType Directory -Force -Path .\data\seedvr2-cache | Out-Null
New-Item -ItemType Directory -Force -Path .\data\seedvr2-output | Out-Null
New-Item -ItemType Directory -Force -Path .\data\seedvr2-tmp | Out-Null
```

## Smoke test (2-3s, VRAM conservateur)

```powershell
docker run --rm --gpus all `
  -v "${PWD}\video:/workspace/input" `
  -v "${PWD}\data\seedvr2-output:/workspace/output" `
  -v "${PWD}\data\seedvr2-cache:/workspace/cache/seedvr2" `
  -v "${PWD}\data\seedvr2-tmp:/workspace/tmp" `
  vast-seedvr2-proto:exp-6a2-v1 `
  --input-video /workspace/input/RS_Negociate_360p.mkv `
  --profile smoke
```

## Test de référence (00:39 -> 00:49, 1280x720)

```powershell
docker run --rm --gpus all `
  -v "${PWD}\video:/workspace/input" `
  -v "${PWD}\data\seedvr2-output:/workspace/output" `
  -v "${PWD}\data\seedvr2-cache:/workspace/cache/seedvr2" `
  -v "${PWD}\data\seedvr2-tmp:/workspace/tmp" `
  vast-seedvr2-proto:exp-6a2-v1 `
  --input-video /workspace/input/RS_Negociate_360p.mkv `
  --profile reference
```

## Vérification sans GPU (dry-run)

```powershell
docker run --rm `
  -v "${PWD}\video:/workspace/input" `
  -v "${PWD}\data\seedvr2-output:/workspace/output" `
  -v "${PWD}\data\seedvr2-cache:/workspace/cache/seedvr2" `
  -v "${PWD}\data\seedvr2-tmp:/workspace/tmp" `
  vast-seedvr2-proto:exp-6a2-v1 `
  --input-video /workspace/input/RS_Negociate_360p.mkv `
  --profile smoke `
  --dry-run
```

## Emplacements des artefacts

Chaque run crée un dossier:

- `data/seedvr2-output/seedvr2-<profil>-<timestamp>-<source>/`

Contenu principal:

- `upscaled_h264_aac.mp4`
- `run.log`
- `prototype-report.json`

## Critères de validation (phase 6A.2)

La faisabilité GPU n'est **pas** considérée validée tant qu'une inférence réelle ne passe pas sur GPU cible.

Go technique minimal:

1. exécution complète sans OOM ;
2. sortie vidéo lisible en 1280x720 (profil reference) ;
3. sortie finale H.264/AAC ;
4. rapport JSON complet (paramètres + checkpoints + timings + statut).

## Opérations requises pour le premier essai Vast.ai (sans exécution ici)

1. Construire l'image locale prototype.
2. Pousser **uniquement après validation explicite** (tag expérimental dédié, jamais stable).
3. Préparer les volumes persistants (cache/output/tmp).
4. Louer manuellement une instance RTX 5080 16 Go.
5. Vérifier sur instance:
   - `nvidia-smi`
   - `torch.cuda.is_available()`
   - nom GPU + capacité.
6. Lancer d'abord `--profile smoke`.
7. Si smoke OK, lancer `--profile reference`.
8. Rapatrier `run.log` + `prototype-report.json` + MP4.
9. Détruire immédiatement l'instance.
10. Décider go/no-go d'intégration applicative à partir des résultats réels.

Commande de référence (instance Vast.ai Linux, image déjà buildée localement):

```bash
docker run --rm --gpus all \
  -v /workspace/video:/workspace/input \
  -v /workspace/data/seedvr2-output:/workspace/output \
  -v /workspace/data/seedvr2-cache:/workspace/cache/seedvr2 \
  -v /workspace/data/seedvr2-tmp:/workspace/tmp \
  vast-seedvr2-proto:exp-6a2-v1 \
  --input-video /workspace/input/RS_Negociate_360p.mkv \
  --profile smoke
```

