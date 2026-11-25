# Roadmap

## État actuel

- Phases 0 à 5: implémentées et validées.
- Phase 6A.2: build SeedVR2 stabilisé (PEP 668), CI dédiée et publication GHCR expérimentale.
- Phase 6A.3: intégration multi-moteur (Real-ESRGAN + SeedVR2 manuel), contrats renforcés, publication image SeedVR2 compatible.
- Phase 6A.4: **en cours (stabilisation orchestration SeedVR2 côté application locale/IHM)**.

## Phase 6A.4 — Stabilisation SeedVR2 (implémentation)

### 6A.4-A. Récupération fiable des résultats distants

- Corriger la découverte des chemins distants SeedVR2 (rapport, vidéo, log).
- Garantir une commande Python distante syntaxiquement valide et correctement échappée.
- Produire un JSON de découverte même sans fichier log.
- Valider strictement les chemins distants et bloquer les sorties hors répertoire autorisé.
- Gérer explicitement: rapport absent, rapport invalide, rapport incomplet.
- Récupérer automatiquement MP4 + rapport JSON + log disponible.

### 6A.4-B. Progression SeedVR2 temps réel (SSE)

- Remonter des étapes réellement observables:
  - préparation locale,
  - transfert entrée,
  - préflight distant,
  - téléchargement/réutilisation checkpoints,
  - encodage VAE,
  - upscaling DiT,
  - décodage VAE,
  - progression chunks/images,
  - encodage final,
  - récupération fichiers.
- Ne pas inventer de pourcentage; utiliser un état indéterminé quand la mesure n’est pas fiable.
- Conserver l’annulation et limiter le volume d’événements SSE.
- Préserver la non-régression Real-ESRGAN.

### 6A.4-C. Fiabilisation des états finaux IHM

- Distinguer explicitement:
  - traitement en cours,
  - inférence réussie / récupération en cours,
  - traitement terminé et fichiers récupérés,
  - inférence échouée,
  - inférence réussie / récupération échouée,
  - traitement annulé.
- Empêcher qu’un message annexe (ex: préservation instance manuelle) écrase l’état final métier.

### 6A.4-D. Cache partagé des checkpoints SeedVR2

- Migrer d’un cache par session vers un cache partagé d’instance (ex: `/workspace/seedvr2-cache`).
- Réutiliser les checkpoints valides avec vérification d’intégrité.
- Gérer téléchargements incomplets et concurrence.
- Séparer cache durable et artefacts job/session.
- Ne pas supprimer le cache lors du nettoyage normal de session.

### 6A.4-E. Récupération post-erreur sans nouvelle inférence

- Permettre de relancer uniquement la récupération des artefacts d’un job SeedVR2 terminé côté distant.
- Conserver les informations minimales de reconnexion/job distant.
- Vérifier l’existence des fichiers et expliciter les erreurs (instance inaccessible, artefacts absents).
- Garantir l’absence de relance d’inférence pendant cette opération.

## Phase 7 — Analyse et restauration pré-upscale (documentation uniquement)

Objectif: améliorer la qualité source avant agrandissement.

- 7A: audit techniques de diagnostic vidéo.
- 7B: analyse automatique des défauts + recommandations.
- 7C: moteur de restauration configurable (FFmpeg + VapourSynth initialement).
- 7D: prévisualisation comparative et intégration IHM.
- 7E: benchmark des gains par moteur d’upscaling.

Défauts cibles: bruit spatial/temporel, macroblocs, ringing, flou, entrelacement, banding, scintillement, contraste, colorimétrie.

Principes:

- diagnostic avec niveau de confiance,
- traitements activables/réglables individuellement,
- prévisualisation avant validation,
- restauration possible sans upscale,
- conservation de la source originale,
- pas de nettoyage agressif systématique.

## Phase 8 — RealBasicVSR (documentation uniquement)

Objectif: proposer un troisième moteur exploitant la cohérence temporelle.

- 8A: audit modèle/licence/checkpoints/dépendances.
- 8B: prototype Docker + compatibilité Blackwell.
- 8C: validation GPU autonome.
- 8D: benchmark comparatif vs Real-ESRGAN/SeedVR2.
- 8E: intégration IHM.

Note: ne pas présumer la compatibilité des anciennes stacks PyTorch/CUDA/MMCV avec GPU récents.

## Phase 9 — Traitement parallèle multi-GPU (documentation uniquement)

Objectif: réduire le temps total d’une même vidéo via multi-GPU sur une instance Vast.ai.

- 9A: audit architecture + ressources partagées.
- 9B: prototype Real-ESRGAN multi-GPU.
- 9C: benchmark 1/2/4 GPU.
- 9D: extension aux moteurs temporels.
- 9E: intégration IHM.

Approche initiale privilégiée: processus indépendants affectés explicitement à chaque GPU, distribution images/segments.

Sujets à traiter:

- détection GPU disponibles,
- sélection du nombre de GPU utilisés,
- affectation explicite des processus,
- équilibrage charge,
- chevauchement segments pour moteurs temporels,
- réassemblage sans perte ni duplication d’images,
- synchronisation audio,
- saturation CPU/RAM/disque,
- gestion des échecs partiels,
- mesure du gain réel et du coût total.

## Chantier transversal — Benchmark (progressif)

Constituer un corpus de référence commun, reproductible.

Premières sources:

- `RS_Negociate_360p.mkv`
- `PB_Bric_360p.mp4`

Règle: ne pas versionner les vidéos dans Git.

Comparaisons cibles:

- source agrandie Lanczos (baseline),
- résultats moteurs (Real-ESRGAN, SeedVR2, futurs moteurs),
- résolution/plage temporelle comparables,
- qualité visuelle + cohérence temporelle,
- temps d’inférence + temps total,
- RAM/VRAM,
- coût de traitement.

## Contraintes transverses permanentes

- Pas d’opération payante Vast.ai sans confirmation explicite.
- Pas de secret commité.
- Pas de transfert vidéo via intermédiaire cloud tiers non maîtrisé.
