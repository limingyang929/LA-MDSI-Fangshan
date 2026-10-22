"""Run unit checks and one real-data logistic-regression validation fold.

This is a functional demonstration, not a substitute for the full analysis.
"""
from pathlib import Path
import argparse
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent / '00_Library'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Run input and formula checks without fitting a model.')
    args = parser.parse_args()
    from project_config import INPUT, FOLD_SOURCE, FEATURES, OUTPUT, ensure_output
    import json
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    import model_core
    import scientific_core as core
    from graph_core import correct_graphs, screen_holm

    data = pd.read_csv(INPUT, float_precision='round_trip').set_index('sample_id', drop=False)
    assert len(data) == 8806 and data.index.is_unique
    assert data.label.value_counts().to_dict() == {1:4403, 0:4403}
    assert np.isfinite(data[FEATURES[20]+['x','y']].to_numpy(float)).all()
    folds = pd.read_csv(FOLD_SOURCE)
    fixed = folds.loc[folds.experiment.eq('main') & folds.seed.eq(42)].set_index('sample_id').loc[data.index]
    assert fixed.fold.nunique() == 5 and fixed.validation_group.nunique() == 69
    assert np.array_equal(data.spatial_group.astype(str), fixed.validation_group.astype(str))
    rng = np.random.default_rng(123)
    xy = rng.normal(size=(80,2))
    graphs, _ = correct_graphs(xy)
    for k, graph in graphs.items():
        assert np.all(graph.diagonal() == 0) and (graph != graph.T).nnz == 0
        assert np.all(np.asarray(graph.sum(axis=1)).ravel() >= k)
    y = np.tile([0,1],40)
    levels = {'tree':np.repeat(np.arange(4),20), 'kmeans':np.arange(80)%5}
    vector = rng.normal(size=80)
    direct_q = 1 - sum(np.sum((vector[levels['tree']==g]-vector[levels['tree']==g].mean())**2) for g in range(4))/np.sum((vector-vector.mean())**2)
    assert np.isclose(core.q_stat(vector,levels['tree']),direct_q)
    assert np.isclose(core.q_stat(vector,levels['tree']),core.q_stat(3*vector+8,levels['tree']))
    for graph in graphs.values():
        assert np.isclose(core.moran(vector,graph),core.moran(3*vector+8,graph))
    values = core.theta_six(y,np.full(80,.5),graphs,levels)
    six = np.array([values[key] for key in core.THETA])
    assert np.isfinite(six).all() and np.max(abs(six)) < 1e-8
    assert np.isclose(values['r_dsi'],np.quantile(six,.25,method='linear'))
    checks = {'input_rows':len(data),'positive_rows':int(data.label.sum()),'fixed_groups':69,
              'graph_checks':True,'formula_checks':True,'purpose':'functional checks, not full reproduction'}
    print(json.dumps(checks,indent=2),flush=True)
    if args.check_only:
        return

    out = ensure_output(OUTPUT/'00_Demonstration')
    train = data.loc[fixed.fold.ne(1)]
    validation = data.loc[fixed.fold.eq(1)]
    cats = ['Landcover','Soil']
    numeric = [key for key in FEATURES[16] if key not in cats]
    processor = ColumnTransformer([('numeric',StandardScaler(),numeric),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cats)])
    weights = np.empty(len(train))
    for label in (0,1):
        mask = train.label.to_numpy()==label
        weights[mask] = len(train)/(2*mask.sum())
    with threadpool_limits(limits=1):
        matrix = processor.fit_transform(train[FEATURES[16]])
        model = model_core.estimator('logistic',43)
        model.fit(matrix,train.label.to_numpy(int),sample_weight=weights)
        probability = model.predict_proba(processor.transform(validation[FEATURES[16]]))[:,1]
        core.FEATURES = FEATURES[16].copy()
        core.CAT = cats
        core.NUM = numeric
        tree, km, _ = core.fit_real_strata(train,validation,43,out/'strata_rules.joblib')
        gs, _ = correct_graphs(validation[['x','y']].to_numpy(float))
        strata = {'tree':tree,'kmeans':km}
        screens = screen_holm(core.identifiability(validation.label.to_numpy(int),gs,strata,499,4201))
        diagnostics = core.theta_six(validation.label.to_numpy(int),probability,gs,strata)
    predictions = validation[['sample_id','label','x','y']].copy()
    predictions['probability'] = probability
    predictions.to_csv(out/'01_Predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(screens).to_csv(out/'02_Response_screen.csv',index=False)
    pd.DataFrame([diagnostics]).to_csv(out/'03_Diagnostics.csv',index=False)
    result = dict(checks,model='logistic',seed=42,fold=1,training_n=len(train),validation_n=len(validation),
                  all_six_tests_pass=all(row['identified'] for row in screens),
                  **model_core.metrics(validation.label.to_numpy(int),probability))
    (out/'04_Run_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    print('Demonstration output:',out)


if __name__ == '__main__':
    main()
