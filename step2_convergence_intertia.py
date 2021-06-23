import numpy as np
import functools

from ema_workbench import (save_results, Constraint, SequentialEvaluator, TimeSeriesOutcome,
                           RealParameter, ScalarOutcome, CategoricalParameter, ema_logging, MultiprocessingEvaluator, perform_experiments)

from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.connectors import vensim
from ema_workbench.em_framework.parameters import Scenario
from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)
#functie van maken!
get_10_percentile = functools.partial(np.percentile, q=10)


def get_inertia(lever):
    inertia = np.sum(np.abs(np.diff(lever)) > 0.02)/float(nvars-1)
    return intertia

def get_last_outcome(outcome,time):
    index = np.where(time == 2100) #model runs until 2100
    last_outcome = outcome[index][0]
    return last_outcome

def get_SD(outcome):
    sd=np.std(outcome)
    return sd

def constraint_biomass(biomass):
    index = np.where(time == 2010)  # model runs from 2010 onwards
    initial_biomass = outcome[index][0]
    lowest_allowable= initial_biomass*0.4
    return lambda biomass:min(0, biomass-lowest_allowable)

def lookup_list(time_range):
    list = np.arange(0, time_horizon, 2).tolist()
    return list

class MyVensimModel(VensimModel):

    def run_experiment(self, experiment):
        # 'Look up harvesting quota'
        # f"Proposed harvesting quota {t}", 0, 10) for t in range(45)]
        lookup = []
        for t in range(45):
            value = experiment.pop(f"Proposed harvesting quota {t}")
            lookup.append((2010+2*t, value))
        experiment['Look up harvesting quota'] = lookup

        return super(MyVensimModel, self).run_experiment(experiment)

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    wd = './Vensim models'
    vensimModel = MyVensimModel("simpleModel", wd=wd,
                               model_file='model_thesis_V50_influence.vpmx')

    # import reference scenario (=worst case scenario from open exploration of base case). even gekopieerd uit notebook
    #ref_scen = open('ref_scen.txt', 'r')
    ref_scen = {'Annual consumption predator': 1.6165677145496105, 'Annual fish consumption per capita': 1.6527319962989554e-05, 'Average sinking time': 327.7211246704461, 'Average weight per adult MF': 1.171647345704505e-06, 'Average weight per juvinile MF': 8.040451308502405e-08, 'Average weight per juvinile predator': 0.082065128959174, 'Average weight per predator': 0.3265896132065439, 'C content copepods': 0.6074742673601248, 'Carbon loss at living depth': 0.2510117441617158, 'Carbon loss underway': 0.029721608696283063, 'Catchability mauriculus': 0.2611190077806341, 'Catchability myctophidae': 0.17596780126184425, 'Consumption by MF in bodyweight': 6.9745451866911194, 'Consumption by zooplankton in bodyweight': 2.7121998161893366, 'Conversion factor to ppm': 2.23560866641554, 'Costs regular fish': 374166494233.4893, 'Costs status quo mauriculus': 351697859312.2777, 'Costs status quo myctophidae': 512901711393.8156, 'Delay sedimentation': 8960.225492500851, 'Delay weathering': 9217.306866065908, 'Depth euphotic zone': 95.84340143188507, 'Downwelling water': 1535050492454873.0, 'Efficiency factor fisheries': 6.369532064204147, 'Export efficiency': 1.0616207453859041, 'Female fraction': 0.5867927720307371, 'Fishmeal to fish factor': 4.191245779130841, 'Fraction grazed C ending up in surface': 0.26753929473497845, 'Fraction of migrating MF constant': 0.3977047661973877, 'Fraction spawning mauriculus vs myctophidae': 0.7442274483201364, 'Grazing in surface by MF': 0.5019664958493991, 'Growth period mauriculus': 1.0140166789485323, 'Growth period myctophidae': 0.6326916272131893, 'Growth period predator': 3.5844916724237805, 'Harvest information delay': 2.8758926223612544, 'Information delay risk perception': 4.372705746915433, 'Initial juvinile predator weight': 6.279783773851887, 'Initial phytoplankton': 1.1215181602817128, 'Initial predator weight': 4.377014207685418, 'Initial sediment C': 3872.690106726809, 'Initial surface C': 575.0533169078308, 'Initial weight mauriculus adult': 7.649815630076603, 'Initial weight mauriculus juvinile': 1.203612725704423, 'Initial weight myctophidae adult': 6.298277804488287, 'Initial weight myctophidae juvinile': 0.8642138620918166, 'Initial zooplankton': 4.546006493615166, 'Life expectancy mauriculus': 2.039092922127697, 'Life expectancy myctophidae adult': 3.96391052320688, 'Living depth mauriculus': 250, 'Living depth myctophidae': 750, 'Other carbon fluxes': 5.570816611278588, 'Other food sources': 1.7531118400179309, 'Percentage discarded fish': 0.0936030802658157, 'Predator life expectancy': 5.023746919347468, 'Residence time deep carbon': 920.8509709192483, 'SWITCH lanternfish to mauriculus': 2, 'Sale price regular fish': 2859505217245.8315, 'Share of aquaculture': 0.5453758190038908, 'Share of irreplaceable fishmeal': 0.8532609598986316, 'Spawning fraction': 0.2048763544926142, 'Specialist capacity building time': 5.37351027161412, 'Surface ocean': 369491291653652.2, 'Survived larvea': 167.7482170989903, 'Switch influence CO2 on phytoplankton': 1, 'Switch influence sunlight on phytoplankton': 1, 'Switch population growth': 3, 'Switch price change': 1, 'Switch profitability change MF fisheries': 2, 'Switch risk perception biomass': 1, 'Switch risk perception climate': 3, 'Switch risk reward mechanism': 2, 'Total atmospheric volume': 3.45813031422299e+18, 'Transfer velocity for GtC per year': 1.1630176695775796, 'Turnover time phytoplankton': 0.08707484999320064, 'Upwelling delay surface': 8.716886449765179, 'ppm conversion for ocean': 2.0475345874735686}
    ref_scen = Scenario('ref_scen', **ref_scen)


    vensimModel.uncertainties = [#structural uncertainties
                                 CategoricalParameter('Switch risk reward mechanism', (1,2,3) ),
                                 CategoricalParameter('SWITCH lanternfish to mauriculus', (1,2) ),
                                 CategoricalParameter('Switch profitability change MF fisheries', (1,2) ),#1
                                 CategoricalParameter('Switch price change', (1,2) ),
                                 CategoricalParameter('Switch risk perception climate', (1,2,3) ),
                                 CategoricalParameter('Switch risk perception biomass', (1,2) ),
                                 CategoricalParameter('Switch influence sunlight on phytoplankton', (1,2) ),
                                 CategoricalParameter('Switch influence CO2 on phytoplankton', (1,2) ),
                                 CategoricalParameter('Switch population growth', (1,2,3) ),
                                 CategoricalParameter("Living depth myctophidae", (550, 650, 750, 850, 950) ),  # 950
                                 CategoricalParameter("Living depth mauriculus", (175, 250, 350, 450) ),  # 350

                                 #parametric uncertainties
                                 RealParameter("Initial weight myctophidae juvinile", 1*0.6 , 1*1.4),  #1 ****
                                 RealParameter("Initial weight mauriculus juvinile", 1*0.6 , 1*1.4),  #0.05
                                 RealParameter("Initial weight myctophidae adult", 9*0.6, 9*1.4),  #9 ** ****
                                 RealParameter("Initial weight mauriculus adult", 9*0.6, 9*1.4),  #0.45
                                 RealParameter("Initial juvinile predator weight", 6*0.8, 6*1.2),  #6
                                 RealParameter("Initial predator weight", 4*0.8, 4*1.2),  #4 ****
                                 RealParameter("Average weight per predator", 0.3*0.8, 0.3*1.2),  #0.3
                                 RealParameter("Average weight per juvinile predator", 0.08*0.8, 0.08*1.2),  #0.08
                                 RealParameter("Initial zooplankton", 4*0.8, 4*1.2),  #4 ** ****
                                 RealParameter("Initial phytoplankton", 1*0.8, 1*1.2),  #1 ****
                                 RealParameter("Initial surface C", 600*0.95,600*1.05),  #600 #CHANGED FROM 20% to 10% ** ***
                                 RealParameter("Initial sediment C", 3390*0.8, 3390*1.2),  #3390 ****
                                 #RealParameter("Initial capacity mixed",50 , 150),  #100
                                 RealParameter("Conversion factor to ppm", 2.0619 * 0.9, 2.0619 * 1.1),  # 2.0619 ****
                                 RealParameter("Surface ocean", 363000000000000*0.9, 363000000000000*1.1),  # 3.63*10^14 #CHANGED FROM 20% to 10% ** *** ****
                                 RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 RealParameter("Fraction grazed C ending up in surface", 0.4*0.6, 0.4*1.4),  # 0.4 **
                                 #RealParameter("Living depth myctophidae", 750, 1050),  # 950
                                 #RealParameter("Living depth mauriculus", 250, 450),  # 350
                                 RealParameter("Other carbon fluxes", 5*0.8, 5*1.2),  # 5
                                 RealParameter("Carbon loss underway", 0.04*0.6, 0.04*1.4),  # 0.04
                                 RealParameter("Carbon loss at living depth", 0.4*0.6, 0.4*1.4),  # 0.4
                                 RealParameter("Average sinking time", 380*0.8, 380*1.2),  # 380
                                 RealParameter("Delay sedimentation", 10000*0.8, 10000*1.2),  # 10000 ****
                                 RealParameter("Delay weathering", 10000*0.8, 10000*1.2),  # 10000
                                 RealParameter("Efficiency factor fisheries", 7 * 0.8, 7 * 1.2), #7
                                 RealParameter("Growth period predator", 3*0.8, 3*1.2),  # 3
                                 RealParameter("Other food sources", 2*0.8, 2*1.2),  # 2
                                 RealParameter("Turnover time phytoplankton", 0.077*0.8, 0.077*1.2),  # 0.077 ****
                                 RealParameter("Consumption by zooplankton in bodyweight", 3*0.8, 3*1.2),  # 3
                                 RealParameter("Harvest information delay", 3*0.8, 3*1.2),  # 3
                                 RealParameter("Costs regular fish", 339000000000*0.8, 339000000000*1.2),  # 3.39e+11
                                 RealParameter("Sale price regular fish", 3100000000000*0.8, 3100000000000*1.2),  # 3.1e+12
                                 RealParameter("Costs status quo myctophidae", 450000000000*0.6, 450000000000*1.4),  # 450*10^9 ****
                                 RealParameter("Costs status quo mauriculus", 300000000000*0.6, 300000000000*1.4),  # 225*10^9
                                 RealParameter("Information delay risk perception", 5*0.8, 5*1.2),  # 5 ****
                                 RealParameter("ppm conversion for ocean", 2.1*0.9, 2.1*1.1),  # 2.1 #CHANGED FROM 20% to 10% *** ****
                                 RealParameter("Downwelling water", 1700000000000000*0.8, 1700000000000000*1.2),  # 1.7*10^15 ***
                                 RealParameter("Residence time deep carbon", 1000*0.9, 1000*1.1),  # 1000 #CHANGED FROM 20% to 10% ***
                                 RealParameter("Upwelling delay surface", 8*0.9, 8*1.1),  # 8 #CHANGED FROM 20% to 10% ***
                                 RealParameter("Share of aquaculture", 0.5*0.8, 0.5*1.2),  # 0.5 ****
                                 RealParameter("Share of irreplaceable fishmeal", 0.8, 1),  # 0.1
                                 RealParameter("Annual fish consumption per capita", 0.000017*0.8, 0.000017*1.2),  # 1.7*10^-5
                                 RealParameter("Percentage discarded fish", 0.08*0.8, 0.08*1.2), #0.08
                                 RealParameter("Specialist capacity building time", 5*0.8, 5*1.2),  # 5
                                 RealParameter("Catchability myctophidae", 0.14*0.6, 0.14*1.4),  # 0.14
                                 RealParameter("Catchability mauriculus", 0.28*0.6, 0.28*1.4),  # 0.28
                                 RealParameter("Fraction of migrating MF constant", 0.37*0.6, 0.37*1.4),  # 0.37 **
                                 RealParameter("Grazing in surface by MF", 0.4*0.6, 0.4*1.4),  # 0.4 ** ****
                                 RealParameter("C content copepods", 0.51*0.8, 0.51*1.2),  # 0.51
                                 RealParameter("Depth euphotic zone", 100*0.9, 100*1.1),  # 100#CHANGED FROM 40% to 25% *** ****
                                 RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 RealParameter("Export efficiency", 0.97*0.6, 0.97*1.4),  # 0.97
                                 RealParameter("Transfer velocity for GtC per year", 1.12169*0.8, 1.12169*1.2),  # 1.12169 ***
                                 RealParameter("Average weight per adult MF", 0.000001*0.8, 0.000001*1.2),  # 1*10^-6 ****
                                 RealParameter("Average weight per juvinile MF", 0.00000007*0.8, 0.00000007*1.2),  # 0.07*10^-6
                                 RealParameter("Life expectancy myctophidae adult", 4*0.8, 4*1.2),  # 4 ****
                                 RealParameter("Life expectancy mauriculus", 2*0.8, 2*1.2),  # 2
                                 RealParameter("Growth period myctophidae", 0.583*0.8, 0.583*1.2),  # 0.583
                                 RealParameter("Growth period mauriculus", 1*0.8, 1*1.2),  # 1
                                 RealParameter("Consumption by MF in bodyweight", 7*0.8, 7*1.2),  # 7 **
                                 RealParameter("Annual consumption predator", 1.6*0.8, 1.6*1.2),  # 1.6
                                 RealParameter("Predator life expectancy", 6*0.8, 6*1.2),  # 6
                                 RealParameter("Survived larvea", 147*0.8, 147*1.2),  # 147
                                 RealParameter("Spawning fraction", 0.18*0.8, 0.18*1.2),  # 0.18 ****
                                 RealParameter("Female fraction", 0.515*0.8, 0.515*1.2),  # 0.515
                                 RealParameter("Fraction spawning mauriculus vs myctophidae", 0.75*0.8, 0.75*1.2),  # 0.75
                                 RealParameter("Fishmeal to fish factor", 4*0.8, 4*1.2)  # 4
                                 ]

    vensimModel.levers = [RealParameter(f"Proposed harvesting quota {t}", 0, 7) for t in range(45)] # 4 goed
    #print(type(vensimModel.levers))

    # !!!
    # decisions = (f"Proposed harvesting quota {t}", 0, 7) for t in range(45)
    # nvars = len(decisions)
    # decisions = np.array(decisions)
    # inertia = np.sum(np.abs(np.diff(decisions)) > 0.02) / float(nvars - 1)

    vensimModel.outcomes = [ScalarOutcome('Average food provision by MF', variable_name='Food provision by MF', kind=ScalarOutcome.MAXIMIZE #namen veranderen naar wat de uitkomsten echt zijn (mean etc)
                                    , function=np.mean),
                            ScalarOutcome('Average vertically migrated carbon', variable_name='Total vertical migration', kind=ScalarOutcome.MAXIMIZE
                                   , function=np.mean),
                            ScalarOutcome('Biomass MF 10th percentile', variable_name='Biomass mesopelagic fish', kind=ScalarOutcome.MAXIMIZE
                                   , function=get_10_percentile),
                            ScalarOutcome('Final atmospheric C level', variable_name=['Atmospheric C', 'TIME'], kind=ScalarOutcome.MINIMIZE
                                          , function=get_last_outcome)
                            #ScalarOutcome('inertia') #!!!
                            ]
    # inertia = np.sum(np.abs(np.diff(decisions)) > 0.02)/float(nvars-1)

    #copied from notebook exploration, /10 & *10. #KLOPT DAT VOOR MINIMA?
    convergence_metrics = [HyperVolume(minimum=[0.0008567,0.0108251,0.0001162,75.7981300], maximum=[162.56582, 149.76262,2291.35864,18607.05700]),
                        EpsilonProgress()]

    #vensimModel.constraints = [Constraint("Average biomass MF", outcome_names="Average biomass MF",
                          #function=constraint_biomass)]

    with SequentialEvaluator(vensimModel) as evaluator:
        results, convergence = evaluator.optimize(nfe=10000, searchover='levers', convergence=convergence_metrics,
                                      epsilons=[0.00001,] * len(vensimModel.outcomes) , Scenario=ref_scen)


    results.to_excel('./Data/results_convergence.xlsx')
    convergence.to_excel('./Data/conv_convergence.xlsx')




