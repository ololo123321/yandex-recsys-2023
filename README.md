1st place solution in Yandex Cup 2023 - ML RecSys (audio classification).

Re-train whole ensemble (train each model on 3 splits of 10-folds kf):
```bash
bash re_train_ensemble.sh
```

Make final submission (inference of each model + blending):
```bash
bash predict_ensemble.sh
```

I have loaded two best submissions:
* `ensembles/ensemble_v30.tsv` - public score: 0.3087, private score: 0.3128.
* `ensembles/ensemble_v31.tsv` - public score: 0.3095, private score: 0.3120. Was selected as final.

Code for models selection: `src/beam_search.py`