import pickle
import numpy as np

# file = 'save_progress_0c6f0540-5200-42ee-8117-15aa2f4ce7ac.pkl'
# file = 'save_progress_0dba1041-9972-4dd6-8399-6d3ef9374743.pkl'
file = 'save_progress_0f0250ad-0ca1-489a-85b3-fc7a5872a124.pkl'
with open(f'temp/{file}', 'rb') as f:
    data = pickle.load(f)

save_test_scores = data['save_test_scores']
save_train_scores = data['save_train_scores']
save_ires = data['save_ires']

print('\nsave_ires_updated:')
print(save_ires)
print('\nsave_test_scores_updated:')
print(save_test_scores)
print('\nsave_train_scores_updated:')
print(save_train_scores)
print()

test_scores_better = [{save_ires_el: save_test_scores_el for save_ires_el, save_test_scores_el in zip(save_ires_case, save_test_scores_case)} for save_ires_case, save_test_scores_case in zip(save_ires, save_test_scores)]

ires_max = 10
CVE_scores = {}
for ires in range(ires_max+1):
    test_scores_ires = [test_scores_better_case[ires] for test_scores_better_case in test_scores_better]
    mean = np.mean(test_scores_ires)
    std  = np.std(test_scores_ires)/np.sqrt(len(test_scores_ires))
    CVE_scores[ires] = {'mean': mean, 'std':std}
    print(f'Nres={ires:3} test score:\t{mean:.2f} ± {std:.2f}')

CVE_min = np.inf
lowest_mean_ires = None
for ires, CVE_score in CVE_scores.items():
    if CVE_score['mean'] < CVE_min:
        CVE_min = CVE_score['mean']
        lowest_mean_ires = ires
limit = CVE_scores[lowest_mean_ires]['mean'] + CVE_scores[lowest_mean_ires]['std']

for ires in range(ires_max+1):
    if CVE_scores[ires]['mean'] < limit:
        selected_Nres = ires
        break
else:
    raise RuntimeError('We shouldn\'t be here')
print(f'Selected Nres Model = {selected_Nres}')