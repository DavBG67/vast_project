# Phase 5A — Audit d’intégration des prétraitements au pipeline Vast.ai

## Périmètre et règle de cette phase

- Cette phase est **audit + conception uniquement**.
- Aucun changement fonctionnel proposé ici n’est implémenté dans le pipeline distant.
- Aucun impact visé sur le comportement actuel sans plan (`balanced` inchangé).

---

## 1) Cartographie technique du pipeline actuel

## Entrées et points d’entrée

- CLI principal: [cli.py](../src/vast_upscaler/cli.py)
  - commande `upscale` → construit `OrchestrationOptions` puis appelle [run_upscale_workflow](../src/vast_upscaler/orchestrator.py).
  - commande `web` → démarre FastAPI via [create_app](../src/vast_upscaler/webapp.py).
- Web local FastAPI: [webapp.py](../src/vast_upscaler/webapp.py)
  - mode automatique: [_default_auto_runner](../src/vast_upscaler/webapp.py)
  - mode manuel: [_default_manual_runner](../src/vast_upscaler/webapp.py)

## Responsabilités orchestrateur local

- Orchestration de bout en bout dans [run_upscale_workflow](../src/vast_upscaler/orchestrator.py):
  1. inspection vidéo source locale (`inspect_video`)
  2. sélection offre Vast + estimation coût
  3. création instance (auto) ou usage instance existante (manuel)
  4. bootstrap SSH/trust
  5. exécution worker distant via [execute_remote_worker](../src/vast_upscaler/remote_worker.py)
  6. transitions d’état session + cleanup auto (mode auto)

## Transfert de fichiers

- Upload input: `transport.upload_file(...)` dans [execute_remote_worker](../src/vast_upscaler/remote_worker.py).
- Download report/output: `transport.download_file(...)` idem.
- Pas de prétraitement intermédiaire implicite dans ce chemin.

## Exécution distante worker

- Commande distante construite dans [_build_worker_command](../src/vast_upscaler/remote_worker.py):
  - `python /app/worker/process.py --input ... --output ... --target-width ... --target-height ... --profile ...`
- Worker runtime: [process.py](../worker/process.py)
  - probe source (ffprobe)
  - `run_streaming_upscale(...)`
  - mux audio/sous-titres/métadonnées (`mux_with_source_streams`)
  - validation output (`validate_output`)
  - rapport JSON (`write_report`)

## Décodage / upscale / redimensionnement / letterbox / encodage / mux

- Décodage: `ffmpeg` en `rawvideo rgb24` depuis flux vidéo source (`-map 0:v:0`) dans [run_streaming_upscale](../worker/process.py).
- Upscale IA: frame par frame via [RealesrganUpscaler.upscale_frame](../worker/pipelines/realesrgan.py).
- Redimensionnement + letterbox: [fit_with_letterbox](../worker/process.py) **après** upscale IA.
- Encodage vidéo final du flux traité: `libx265` (CRF configurable, preset medium) dans [run_streaming_upscale](../worker/process.py).
- Mux final:
  - primaire: vidéo traitée + audio `1:a?` + sous-titres `1:s?` + metadata + chapitres
  - fallback sans sous-titres en cas d’échec
  - via [mux_with_source_streams](../worker/process.py).

## Téléchargement résultat et rapports

- Worker écrit `report.json` distant puis sortie vidéo.
- Côté local, [remote_worker.py](../src/vast_upscaler/remote_worker.py):
  - télécharge et valide report (`status=completed`, `frames_processed>0`, path attendu)
  - télécharge vidéo
  - valide dimensions/durée via ffprobe local.

---

## 2) Audit résolution et charge GPU

## Faits de code

- Canvas cible: [compute_target_canvas](../worker/process.py) retourne aujourd’hui les dimensions cibles demandées (pas d’adaptation dynamique).
- Modèle IA:
  - `model_scale=4`
  - `tile=256`, `tile_pad=10`
  - définis par [select_model](../worker/models.py), utilisés dans [RealesrganUpscaler.load](../worker/pipelines/realesrgan.py).
- Ordre réel:
  1. decode frame source
  2. upscale IA (`outscale=4`)
  3. resize/letterbox vers canvas cible
  4. encode vidéo finale

## Distinction des volumes de pixels

- **Pixels source décodés**: dimensions de la vidéo en entrée du worker.
- **Pixels traités par le modèle**: dimensions source frame-by-frame (coût IA principal).
- **Pixels produits par le modèle**: ~ `(source_w*4) x (source_h*4)`.
- **Pixels canvas final**: `target_width x target_height` (après `fit_with_letterbox`), surtout coût CPU encode/resize.

## Question explicite: sortie cible 1280×720, crop source 640×480 → 628×408

### Conclusion

**Oui, ce crop peut réduire le calcul Real-ESRGAN** si appliqué **avant** l’appel au modèle.

- Source pixels:
  - 640×480 = 307 200
  - 628×408 = 256 224
  - réduction ≈ **16,6%**
- Sortie brute du modèle (`scale=4`):
  - 2560×1920 = 4 915 200
  - 2512×1632 = 4 099 584
  - réduction ≈ **16,6%**
- Canvas final reste 1280×720, donc **ne reflète pas** directement la charge IA.

### Nuance importante

- Si le crop est injecté dans la chaîne ffmpeg de décodage côté distant, la vidéo compressée est toujours décodée, puis filtrée/croppée avant passage Python.
- Le gain principal recherché est bien sur la charge IA (entrée modèle plus petite), pas sur la seule taille du canvas final.

## Points restant à mesurer réellement

- Relation exacte coût GPU vs réduction pixels selon scènes et mémoire/tile/OOM.
- Impact CPU côté worker d’un filtrage ffmpeg (`crop`, `bwdif`) ajouté avant transfert vers Python.
- Impact temps total (decode+filters+IA+encode) sur plusieurs classes de contenus.

---

## 3) Comparaison architectures A / B / C

## Option A — Prétraitement local avant transfert

### Avantages

- Réutilise [local_preprocessing.py](../src/vast_upscaler/local_preprocessing.py) déjà disponible.
- Réduction potentielle des pixels envoyés au worker (si crop) et donc charge IA.
- Contrôle local explicite, reproductible, sans modifier le worker.

### Inconvénients

- Réencodage intermédiaire avec perte potentielle qualité.
- Coût CPU local + temps local supplémentaire.
- Fichier intermédiaire potentiellement volumineux à transférer.
- Double pipeline ffmpeg (local puis distant).

## Option B — Prétraitement distant dans le décodage worker

### Avantages

- Évite le réencodage intermédiaire local.
- Un seul encodage final (celui du worker).
- Réduction potentielle directe des pixels IA si crop appliqué avant `upscale_frame`.

### Inconvénients

- Nécessite extension contrat plan→worker.
- Complexifie la validation et l’audit sécurité côté worker (filtres, paramètres).
- Charge CPU distante augmente (filtres ffmpeg supplémentaires).
- Doit préserver strictement compatibilité sans plan.

## Option C — Hybride (choix d’emplacement explicite)

### Avantages

- Peut minimiser réencodage intermédiaire tout en gardant option locale.
- Permet de choisir en fonction du contexte (CPU local disponible, taille upload, coût GPU, risque qualité).

### Inconvénients

- Complexité produit et test plus élevée.
- Nécessite une politique de sélection claire et explicite (jamais implicite).
- Maintenance plus coûteuse (deux chemins d’exécution).

## Synthèse comparative

- Complexité: A < B < C (sur l’intégration worker), mais C > B > A globalement.
- Qualité potentielle: B meilleur qu’A (évite réencodage intermédiaire), C variable selon choix.
- Performance/coût: B/C potentiellement meilleurs pour GPU si crop activé, A peut perdre localement en CPU+temps.
- Maintenance: A plus simple court terme, B/C plus robustes long terme si contrat propre.

---

## 4) Contrat minimal versionné planificateur → worker (proposition)

Objectif: transmettre **uniquement** l’exécutable, pas les heuristiques.

```json
{
  "version": "1.0",
  "operations": [
    {
      "id": "crop",
      "enabled": true,
      "params": { "x": 0, "y": 20, "width": 628, "height": 408 }
    },
    {
      "id": "deinterlace",
      "enabled": false,
      "params": { "mode": "bwdif", "parity": "auto", "deint": "all" }
    }
  ]
}
```

Règles côté worker (proposées):

- version obligatoire, mismatch => erreur explicite.
- opérations inconnues => refus explicite.
- `enabled=false` => ignoré, jamais activé implicitement.
- validation stricte bounds crop + parité + dimensions.
- sans plan => comportement strictement identique actuel.
- journaliser `requested_operations` vs `applied_operations` dans le report worker.

---

## 5) Compatibilité et sécurité (exigences à préserver)

- `balanced` sans plan: inchangé.
- aucun prétraitement implicite.
- modes auto et manuel Vast.ai inchangés (gestion location/destruction intacte).
- paramètres ffmpeg construits en listes d’arguments (pas de `shell=True`).
- validation stricte rectangles crop et paramètres bwdif.
- erreurs explicites en cas de plan incompatible.

---

## 6) Plan d’implémentation incrémental (proposé)

## Étape 1 — Contrat transporté jusqu’au worker, sans activation

- Modules:
  - [remote_worker.py](../src/vast_upscaler/remote_worker.py)
  - [worker/process.py](../worker/process.py)
- Ajout:
  - option `--preprocessing-plan-json` (ou fichier) acceptée mais ignorée si vide.
  - parsing + validation version seulement.
- Tests:
  - compatibilité rétro sans plan.
  - plan invalide/refus explicite.
- Risque:
  - faible si feature-gated.

## Étape 2 — Application distante de `crop` dans decode ffmpeg

- Modules:
  - [worker/process.py](../worker/process.py)
- Ajout:
  - construire filtre decode `crop=...` avant sortie rawvideo.
  - recalcul dimensions source effectives post-crop.
- Tests:
  - résolution effective utilisée par IA.
  - no-op quand crop disabled.
- Risque:
  - erreur de validation dimensions ou pixel format.

## Étape 3 — Application distante explicite `bwdif`

- Modules:
  - [worker/process.py](../worker/process.py)
- Ajout:
  - chaîne filtres decode ordonnée (`bwdif`, puis éventuel crop ou inverse selon choix justifié).
- Tests:
  - `enabled=false` no-op.
  - `enabled=true` appliqué.
- Risque:
  - coût CPU distant / stabilité cadence.

## Étape 4 — Reporting enrichi opérations demandées/appliquées

- Modules:
  - [worker/process.py](../worker/process.py)
  - [remote_worker.py](../src/vast_upscaler/remote_worker.py)
- Ajout:
  - report worker: `requested_operations`, `applied_operations`, `plan_version`.
- Tests:
  - cohérence report + validations côté local.
- Risque:
  - contrat JSON report à aligner proprement.

## Étape 5 — Intégration CLI/Web optionnelle vers chemin distant

- Modules:
  - [cli.py](../src/vast_upscaler/cli.py)
  - [webapp.py](../src/vast_upscaler/webapp.py)
  - [orchestrator.py](../src/vast_upscaler/orchestrator.py)
- Ajout:
  - option explicite pour transmettre un plan déjà validé.
- Tests:
  - mode auto + mode manuel.
  - absence de plan => exact legacy behavior.
- Risque:
  - régression flux orchestration si options mal propagées.

---

## 7) Architecture recommandée

### Recommandation principale

**Option B** comme cible primaire (prétraitement distant intégré au décodage worker), avec **Option A conservée** pour cas opérateur avancé.

Justification:

- évite réencodage intermédiaire local (qualité + temps + stockage)
- permet réduction de charge IA via crop avant modèle
- garde un seul encodage final et un seul artefact vidéo principal.

### Garde-fous

- activation uniquement via plan explicite.
- aucun fallback silencieux d’opérations inconnues.
- rapport explicite demandé/appliqué.
- feature flag / option opt-in initiale pour limiter risque de régression.

---

## Risques et points à mesurer en conditions réelles

- Gain wall-clock réel de crop distant selon scènes et GPU.
- Surcoût CPU de `bwdif`/`crop` côté worker.
- Effet qualité global (sans benchmark historique altéré):
  - comparaison A/B reproductible sur mêmes entrées:
    1. pipeline actuel (baseline)
    2. pipeline avec crop distant
    3. pipeline local prétraité + distant baseline
- Vérifier robustesse sur contenus:
  - sous-titres incrustés
  - changements de ratio en cours de vidéo
  - cadence mixte.

---

## Conclusion

Le code actuel permet déjà de conclure que la charge Real-ESRGAN dépend des dimensions d’entrée modèle, pas du canvas final.  
Un crop appliqué avant modèle peut réduire le coût IA même avec une sortie cible fixe 1280×720.  
L’intégration recommandée est distante (Option B), avec contrat minimal versionné et activation strictement explicite.
