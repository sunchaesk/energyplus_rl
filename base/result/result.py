
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("white")
sns.set_style('darkgrid',{"axes.facecolor": ".92"}) # (1)
sns.set_context('notebook')

caps_scores = [
    4511.812799679611,
    4462.905918533585,
    4443.124962536455,
    4454.459629698134,
    4490.325713717578,
    4408.599617312693,
    4430.639940864781,
    4446.191143879814,
    4454.658839379249,
    4482.110175327222,
    4436.578002441756,
    4492.207266496328,
    4439.567906270799,
    4460.871518728789,
    4468.535152537734,
    4463.478096355009,
    4455.791406971911,
    4452.202026650554,
    4449.388648501366,
    4418.926131156728,
    4472.536147025204,
    4459.607469883801,
    4421.215700111116,
    4470.536282406871,
    4441.871494275801,
    4447.089332227671,
    4448.357635680144,
    4440.329038109882,
    4431.906052564649,
    4446.701317700236,
    4462.329185179543,
    4436.962350429042,
    4446.890962904533,
    4472.93334363398,
    4452.590062234084,
    4450.014129703534,
    4444.738022688768,
    4450.33365255445,
    4463.273156343159,
    4450.914094379106,
]

caps_total_variance = [
    760.100000000002,
    755.1000000000015,
    758.3000000000017,
    791.1000000000013,
    783.7000000000021,
    780.3000000000017,
    768.1000000000016,
    773.5000000000017,
    777.1000000000017,
    756.1000000000017,
    804.1000000000016,
    743.9000000000017,
    773.7000000000019,
    758.7000000000014,
    779.7000000000014,
    762.1000000000018,
    771.5000000000015,
    781.9000000000012,
    790.5000000000015,
    779.9000000000015,
    775.7000000000016,
    756.1000000000016,
    775.3000000000015,
    750.5000000000015,
    779.5000000000018,
    773.5000000000014,
    788.7000000000013,
    766.1000000000017,
    799.5000000000014,
    778.3000000000013,
    760.3000000000018,
    797.1000000000015,
    737.7000000000016,
    772.9000000000021,
    786.1000000000014,
    772.9000000000015,
    783.3000000000014,
    768.9000000000016,
    786.5000000000022,
    769.3000000000023,
]

caps_avg_pmv = [
    0.5867260156431773,
    0.5816700918998619,
    0.5814649050276329,
    0.5746038447753281,
    0.5828387000700942,
    0.5731509801629424,
    0.5708656610059198,
    0.5834653661431117,
    0.5768584340779325,
    0.5790277717646657,
    0.5817868924396552,
    0.594102172444607,
    0.5824861785196035,
    0.5864893494060909,
    0.5858502063961163,
    0.5835261397979087,
    0.5814419718137939,
    0.5860978079393111,
    0.573490651909736,
    0.5739466250587856,
    0.5796167504056251,
    0.5771215984814535,
    0.5731120579891512,
    0.5862515803579472,
    0.5864356639808214,
    0.5799535655679628,
    0.5807868013392699,
    0.5821908434246784,
    0.569305253012219,
    0.5804575807435012,
    0.5875680756992621,
    0.5659312259704871,
    0.5845366366464376,
    0.5824399335694762,
    0.5820907528737449,
    0.5763227459843483,
    0.5744846682540394,
    0.5714409552714514,
    0.5910875195640366,
    0.5927443635209844,
]

max_score = 5426.59396
max_total_variance = 0
max_avg_pmv = 0.847535

no_caps_score = 4377
max_no_caps_score = 4444
min_no_caps_score = 4331
tot_variance = 873
no_caps_total_variance = 830
min_total_variance = 775

max_no_caps_pmv = 0.611
no_caps_avg_pmv = 0.601
min_no_caps_pmv = 0.58

def avg(l:list):
    return (sum(l) / (len(l)))

# plan
# plot 1: average score
# plot 2: average pmv
# plot 3: total variance

def total_variance():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams['errorbar.capsize'] = 20

    algorithms = ["RL + Action Smoothing", "RL"]
    colour = ['red', 'blue']
    scores = [avg(caps_total_variance), no_caps_total_variance]
    max_scores = [max(caps_total_variance), tot_variance]
    min_scores = [min(caps_total_variance), min_total_variance]

    df = pd.DataFrame({'HVAC Control Strategy': algorithms, 'Total Variance $\Sigma |u_{i+1} - u_i|$': scores, 'mmin': min_scores, 'mmax': max_scores})
    df['ymin'] = [scores[i] - min_scores[i] for i in range(len(scores))]
    df['ymax'] = [max_scores[i] - scores[i] for i in range(len(scores))]
    yerr = df[['ymin', 'ymax']].T.to_numpy()

    ax = sns.barplot(x='HVAC Control Strategy', y='Total Variance $\Sigma |u_{i+1} - u_i|$', data=df, yerr=yerr, ax=ax, edgecolor='black')
    ax.set_xlabel('HVAC Control Strategy', fontsize=20)
    ax.set_ylabel('Total Variance $\Sigma |u_{i+1} - u_i|$', fontsize=20)

    plt.title('Total Variance ($\Sigma |u_{i+1} - u_i|$) for each HVAC Control Strategy', fontsize=20)
    plt.tight_layout()
    plt.savefig("total_variance.png", dpi=300, bbox_inches='tight')
    plt.show()

def avg_pmv():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams['errorbar.capsize'] = 20

    colour = ['red', 'blue', 'green']
    algorithms = ["RL + Action Smoothing", "RL", "max setpoint"]
    scores = [avg(caps_avg_pmv), no_caps_avg_pmv, max_avg_pmv]
    max_scores = [max(caps_avg_pmv), max_no_caps_pmv, max_avg_pmv]
    min_scores = [min(caps_avg_pmv), min_no_caps_pmv, max_avg_pmv]

    df = pd.DataFrame({'HVAC Control Strategy': algorithms, 'Thermal Comfort Value (PMV)': scores, 'mmin': min_scores, 'mmax': max_scores})
    df['ymin'] = [scores[i] - min_scores[i] for i in range(len(scores))]
    df['ymax'] = [max_scores[i] - scores[i] for i in range(len(scores))]
    yerr = df[['ymin', 'ymax']].T.to_numpy()

    ax = sns.barplot(x='HVAC Control Strategy', y='Thermal Comfort Value (PMV)', data=df, yerr=yerr, ax=ax, edgecolor='black')
    ax.set_xlabel('HVAC Control Strategy', fontsize=20)
    ax.set_ylabel('Thermal Comfort Value (PMV)', fontsize=20)

    plt.title('Average PMV value for each HVAC control strategy', fontsize=20)
    plt.tight_layout()
    plt.savefig("avg_pmv.png", dpi=1000, bbox_inches='tight')
    plt.show()

def avg_score():
    fig, ax = plt.subplots(figsize = (8,6))
    plt.rcParams['errorbar.capsize'] = 20
    x_pos = [0.1,0.1]
    colour = ["red", "blue", "green"]
    algorithms = ["RL + Action Smoothing", "RL", "max setpoint"]
    scores = [avg(caps_scores), no_caps_score, max_score]
    max_scores = [max(caps_scores), max_no_caps_score, max_score]
    min_scores = [min(caps_scores), min_no_caps_score, max_score]

    df = pd.DataFrame({'HVAC Control Strategy': algorithms, 'Cost (¢) from period 6/21 ~ 6/28': scores, 'mmin': min_scores, 'mmax': max_scores})
    df['ymin'] = [scores[i] - min_scores[i] for i in range(len(scores))]
    df['ymax'] = [max_scores[i] - scores[i] for i in range(len(scores))]
    yerr = df[['ymin', 'ymax']].T.to_numpy()
    ax = sns.barplot(x='HVAC Control Strategy', y='Cost (¢) from period 6/21 ~ 6/28', data=df, yerr=yerr, ax=ax, edgecolor='black')
    ax.set_xlabel('HVAC Control Strategy', fontsize=20)
    ax.set_ylabel('Cost (¢) from period 6/21 ~ 6/28', fontsize=20)

    #ax.bar(algorithms, scores, yerr=yerr, capsize=20, color=colour, width=0.3)
    #plt.legend()
    plt.title('Cost of operation for each HVAC control strategy', fontsize=20)
    plt.tight_layout()
    plt.savefig("avg_score.png", dpi=1000, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # print('avg:', avg(caps_scores), avg(caps_total_variance), avg(caps_avg_pmv))
    # print('max:', max(caps_scores), min(caps_scores))
    # print('avg', avg(caps_avg_pmv), 'max', max(caps_avg_pmv), 'min', min(caps_avg_pmv))
    print(max(caps_total_variance), min(caps_total_variance), avg(caps_total_variance))
    avg_score()
    avg_pmv()
    total_variance()

# intro
# 1. find a way to interest (hook)
# - hook
#   - buildings take up 30% of energy, 50% of which is spent for HVAC
# - identify preliminary research question, and research gap
#   - traditional rule-based HVAC controllers fail to fully explore the
#     controller space and utilize the growing availability of building operational
#     and recent advancements in data-driven control/optimization methods
# 2. identify the structure and organizational logic
# 3. main claim
