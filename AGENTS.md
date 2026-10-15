# ASTRA CONSIGNES Ω — Non-répétition (projet splatforge-platform)

Ces règles ont été explicitement fixées par le propriétaire. Elles s'appliquent aux projets antérieurs, actuels et futurs. Source de vérité versionnée : dépôt privé `Dr-starck66/indexation`, fichiers `config/astra-consignes.json` et `tools/astra-consignes/`, ainsi que `AGENTS.md`. Lire le contrat actuel AVANT de planifier, modifier ou déployer ; si ce dépôt n'est pas accessible, utiliser les invariants ci-dessous et marquer la vérification de la version centrale UNVERIFIED, ne pas inventer la validation.

## Invariants transversaux
1. **Zéro dépense supplémentaire / abonnement / API payante sans consentement exprès** ; optimiser les capacités et connecteurs déjà existants.
2. **Ne pas créer de tâche d'automatisation supplémentaire : fusionner la mission à un batch actif existant**. Ne jamais supprimer ou écraser une sous-mission indépendante.
3. **Quand il s'agit d'un site éditorial de production : 3 nouvelles publications réelles PAR JOUR et PAR SITE**, actualités vérifiées d'abord, sinon evergreen SEO original et pertinent ; une seule publication est insuffisante. Contrôler 3 URLs publiques distinctes et leur date réelle. Ceci n'impose PAS des articles aux bibliothèques et applications non éditoriales.
4. **Zéro faux PASS** : preuve indépendante code → tests → déploiement du bon SHA → domaine HTTPS, contenu réel, URL et image, et séparation de l'éventuelle présence sur Google Discover.
5. **Rechercher les consignes existantes avant d'interroger l'utilisateur** ; préserver décisions plus anciennes tant qu'elles ne sont pas explicitement révisées. Toute contradiction est bloquée et remontée sans réécriture silencieuse.
6. Pour WINE ELSE, périmètre **toute l'Occitanie**, pas le seul Roussillon ou Languedoc.
7. Préserver les règles spécifiques au projet, ses URL historiques, données et identifiants ; ne pas modifier aveuglément ses déploiements, secrets ou DNS.

## Contrôle de conformité
Le garde exécutable existe dans `Dr-starck66/indexation/tools/astra-consignes/cli.mjs`. Toute release outillée doit effectuer une vérification de plan et conserver l'empreinte du contrat. La présence de ce fichier constitue un **raccordement documentaire**, pas une preuve que tous les pipelines exécutent ce garde ni que les quotas de production sont déjà atteints. Signaler distinctement ce qui reste UNVERIFIED. Aucun coût supplémentaire et aucune création de nouvelle tâche.
