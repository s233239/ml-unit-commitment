import sys

import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pandas import DataFrame

warnings.simplefilter(action='ignore')

# Plot settings
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams["axes.grid"] = True
plt.rcParams['grid.linestyle'] = '-.'
plt.rcParams['grid.linewidth'] = 0.4


class unit_comitment():
    demand: DataFrame | None

    def __init__(self):

        ## Uploading the input data
        # B-matrix
        self.B = pd.read_csv('./input/B (power transfer factor of each bus to each line).csv', sep=';')

        # Capacity of wind farms
        self.wfc = pd.read_csv('./input/Capacity of wind farms.csv', sep=';')['wind'].tolist()
        # Loads
        self.loads = pd.read_csv('./input/Loads.csv', sep=';')['load'].tolist()
        # Pmax of conv gens
        self.pg_max = pd.read_csv('./input/Maximum production of generating units.csv', sep=';')['pgmax'].tolist()
        # Pmin of conv gens
        self.pg_min = pd.read_csv('./input/Minimum production of generating units.csv', sep=';')['pgmin'].tolist()
        # Min gen downtime
        self.pg_min_downtimes = pd.read_csv('./input/Minimum down time of generating units.csv', sep=';')['ld'].tolist()
        # Min gen uptime
        self.pg_min_uptimes = pd.read_csv('./input/Minimum up time of generating units.csv', sep=';')['lu'].tolist()
        # Production costs of gen units
        self.g_costs = pd.read_csv('./input/Production cost of generating units.csv', sep=';')['cost_op'].tolist()
        # Ramping rate of gen units
        self.ramp_rates = pd.read_csv('./input/Ramping rate of generating units.csv', sep=';')['ru'].tolist()
        # Start-up costs
        self.startup_costs = pd.read_csv('./input/Start-up cost of generating units.csv', sep=';')['cost_st'].tolist()
        # Line transmission capacities
        self.transmission_caps = pd.read_csv('./input/Transmission capacity of lines.csv', sep=';')['fmax'].tolist()

        # WF, gen, and load position in the system. starts with 0!
        self.gen_idx = [0, 1, 5] #-> node idx where there is a conv gen
        self.wf_idx = [3, 4]
        self.ld_idx = [3, 4, 5]

        # Defining the lengths of the sets we are working with
        self.T = 24  # time span in hrs
        self.N = 6 # Number of busses in the system
        self.GC = len(self.gen_idx) # conv generator number
        self.WF = len(self.wf_idx)  # Number of wind farms
        self.G = self.GC + self.WF # all generators

        self.L = len(self.loads)  # Loads
        self.lines = len(self.transmission_caps) # total number of lines in the system

        self.M = 1e4 # penalization term for the slack variables in the objective function

        # wind data
        self.wind_samples = pd.read_csv('./input/processed/wind_samples.csv', sep=',', decimal='.')
        # Load data
        self.load_samples = pd.read_csv('./input/processed/load_samples.csv', sep=',', decimal='.')

        # wind turbine 24hr-production forecast
        self.wf_power = pd.read_csv('./input/wind_data.csv', sep =';')

        # system demand over 24 hrs on each bus:
        self.demand = pd.read_csv('./input/demand_data.csv', sep=';')

        # mapping generators & WF to nodes: a dict of nodes, each pointing to a list of generators, connected there
        # all are zero indexed !
        # nodes : [0 ... 6] ; gen : [0..3]
        # {node0 : [gen0]}
        self.gen_nodes = {0:[0], 1:[1], 2:[] , 3:[3], 4:[4], 5:[2]}
        self.wind_nodes = {0:[], 1:[], 2:[] , 3:[0], 4:[1], 5:[]}
        self.load_nodes = {0:[], 1:[], 2:[] , 3:[3], 4:[4], 5:[5]}

        # Status of conv. gens @ T = -1 : are the gens on before the start of the model
        a = 1
        self.u1_0 = a
        self.u2_0 = a
        self.u3_0 = a
        self.u4_0 = a
        self.u5_0 = a

    def print_items(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}:\n {value}\n")

    def solve_model(self):
        ## Set up the model

        direction = gb.GRB.MINIMIZE  # Min the cost
        m = gb.Model()  # Create a Gurobi model
        m.setParam('OutputFlag', 0)  # 1 == detailed output for debugging

        # ========================================= Decision Variables ============================================
        p = m.addVars(self.G, self.T, lb=0, ub=gb.GRB.INFINITY, name="p")  ## gen / wf g dispatch at time t
        s = m.addVars(self.G, self.T, lb=0, ub=gb.GRB.INFINITY, name="s")
        u = m.addVars(self.G, self.T, vtype=gb.GRB.BINARY, name="u") # binary variables for on/off status of generators

        # Slacks:
        e = m.addVars(self.N, self.T, lb=0, ub=gb.GRB.INFINITY, name="e")
        d = m.addVars(self.N, self.T, lb=0, ub=gb.GRB.INFINITY, name="d")

        # ========================================= Objective Function ============================================
        obj = gb.quicksum( (self.g_costs[g] * p[g, t] + s[g, t]) for g in range(self.G) for t in range(self.T)) \
              + self.M*gb.quicksum(e[n, t] + d[n, t] for n in range(self.N) for t in range(self.T)) ## WF costs are 0

        m.setObjective(obj, direction)

        # ========================================= Balancing Equation ============================================
        for t in range(self.T):
            m.addConstr((gb.quicksum(p[g, t] for g in range(self.G)) ==
            gb.quicksum(self.demand.iloc[t, n] + e[n, t] - d[n, t] for n in range(self.N))), name=f'BALANCE_{t}')

        # ======================================== Generator limits ==============================================
        m.addConstrs(u[g, t] * self.pg_min[g] <= p[g, t] for g in range(self.G) for t in range(self.T))
        m.addConstrs(u[g, t] * self.pg_max[g] >= p[g, t] for g in range(self.G) for t in range(self.T))

        m.addConstrs((p[self.GC + g, t] <= self.wf_power.iloc[t, g] for g in range(self.WF) for t in range(self.T)), name='Wind')

        # ========================================= Line Flow Limits ==============================================
        for t in range(self.T): # for each time
            for n in range(self.N): # for each node
                for l in range(self.lines): # for each line
                    m.addConstr(gb.quicksum(self.B.iloc[l, n] * (p[g, t] - self.demand.iloc[t, n] - e[n, t] + d[n, t])
                                            for g in self.gen_nodes.get(n)) <= self.transmission_caps[l], name = f'FLow_{t}{n}{l}_max')
                    m.addConstr(gb.quicksum(self.B.iloc[l, n] * (p[g, t] - self.demand.iloc[t, n] - e[n, t] + d[n, t])
                                            for g in self.gen_nodes.get(n)) >= -self.transmission_caps[l], name = f'FLow_{t}{n}{l}_min')

        # ========================================= Startup Cost Constr. ===========================================
        m.addConstr(u[0, 0] == self.u1_0)
        m.addConstr(u[1, 0] == self.u2_0)
        m.addConstr(u[2, 0] == self.u3_0)
        m.addConstr(u[3, 0] == self.u4_0)
        m.addConstr(u[4, 0] == self.u5_0)

        m.addConstrs(s[g, t] >= self.startup_costs[g] * (u[g, t] - u[g, t-1]) for g in range(self.G) for t in range(1, self.T))
        m.addConstrs(s[g, t] >= 0 for g in range(self.G) for t in range(self.T))

        # ========================================= Ramping Constr. ================================================
        m.addConstrs(p[g, t] - p[g, t-1] <= self.ramp_rates[g] for g in range(self.G) for t in range(1, self.T))
        m.addConstrs(p[g, t-1] - p[g, t] <= self.ramp_rates[g] for g in range(self.G) for t in range(1, self.T))

        # ========================================= Max Up/Down Time =============================================
        for g in range(self.G):
            for t in range(1, self.T):
                for tau in range(1, min(t + 1, self.pg_min_uptimes[g])):
                    m.addConstr(u[g, t - tau] + u[g, t] - u[g, t - tau + 1] <= 1)
                for tau in range(1, min(t + 1, self.pg_min_downtimes[g])):
                    m.addConstr(-u[g, t - tau] + u[g, t] + u[g, t - tau + 1] <= 1)

        #================================================ Solve  =============================================
        m.update()
        m.write('unit_comittment_model.lp')
        m.optimize()

        # Satus
        self.print_status(m)

        #================================================ Results  =============================================
        columns = []
        columns.append('Time')

        # Record dispatched power
        for g in range(self.G):
            columns.append(f'gen{g+1}')
        columns.append('total_load')
        columns.append('total_dispatch')

        # Record startup costs
        for g in range(self.G):
            columns.append(f's{g+1}')

        # Record binaries
        for g in range(self.G):
            columns.append(f'u{g+1}')

        # Record slacks
        for n in range(self.N):
            columns.append(f'e{n + 1}')
        for n in range(self.N):
            columns.append(f'd{n + 1}')

        results = pd.DataFrame(columns=columns)

        ## Record for each t
        t_result = {}
        for t in range(self.T):
            t_result['Time'] = t
            # record dispatch
            sum = 0
            for g in range(self.G):
                t_result[f'gen{g+1}'] = p[g, t].x
                sum += p[g, t].x
            # sum of dispatch
            t_result['total_dispatch'] = sum
            # record load @ t
            t_result['total_load'] = self.demand.iloc[t,:].sum()
            # s
            for g in range(self.G):
                t_result[f's{g+1}'] = s[g,t].x
            for g in range(self.G):
                t_result[f'u{g+1}'] = u[g,t].x
            # record slack
            for n in range(self.N):
                t_result[f'e{n+1}'] = e[n, t].x
            for n in range(self.N):
                t_result[f'd{n + 1}'] = d[n, t].x

            results.loc[len(results)] = t_result

        return results

    def plot_results(self, results:pd.DataFrame, hour:int):

        disp = results[['gen1', 'gen2', 'gen3', 'gen4', 'gen5']].copy(deep=True)
        all = results[['gen1', 'gen2', 'gen3', 'gen4', 'gen5', 'total_load']].copy(deep=True)
        capacity = self.pg_max

        # WF max power @ hout t is the forecast. The WFs can be down-regulated, if necessary
        capacity[3] = self.wf_power.iloc[hour, 0]
        capacity[4] = self.wf_power.iloc[hour, 1]

        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3), dpi=200)
        fig.suptitle(f'Optimal Dispatch. t = {hour}')

        df = disp.T
        total_load = all['total_load'][hour].round(2)
        ax1.bar(df.index, df.iloc[:,hour], label='Gen_Dispatch', width=0.4)
        ax1.bar(df.index, capacity, label = 'Max Capacity &\nPredicted WF Power', width = 0.4, edgecolor='black', linestyle='--', fill=False)
        ax1.text(x=3, y=150, s=f'Total load={total_load}MW', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='blue', alpha=0.5))

        ax1.set_xticklabels(['G1', 'G2', 'G3', 'WF1', 'WF2'])
        ax1.set_ylabel('Dispatched Power [MW]')
        ax1.legend(fontsize = 7)

        plt.savefig(f'./figures/opt_dispatch_t={hour}.png')
        plt.show()


    def print_status(self, m):
        status = m.status
        if status == gb.GRB.OPTIMAL:
            print("Optimal solution found:", m.objVal)
        elif status == gb.GRB.INFEASIBLE:
            print("Model is infeasible")
        elif status == gb.GRB.UNBOUNDED:
            print("Model is unbounded")
        elif status == gb.GRB.ITERATION_LIMIT:
            print("Optimization terminated because the total number of simplex iterations performed exceeded the limit")
        elif status == gb.GRB.NODE_LIMIT:
            print(
                "Optimization terminated because the total number of branch-and-cut nodes explored exceeded the limit")
        elif status == gb.GRB.TIME_LIMIT:
            print("Optimization terminated because the time expended exceeded the limit")
        elif status == gb.GRB.SOLUTION_LIMIT:
            print("Optimization terminated because the number of solutions found reached the limit")
        elif status == gb.GRB.INTERRUPTED:
            print("Optimization was terminated by the user")
        elif status == gb.GRB.NUMERIC:
            print("Optimization was terminated due to unrecoverable numerical difficulties")
        elif status == gb.GRB.SUBOPTIMAL:
            print("Unable to satisfy optimality tolerances; a sub-optimal solution is available")
        elif status == gb.GRB.INPROGRESS:
            print("Asynchronous optimization call was made, but the associated optimization run is not yet complete")
        elif status == gb.GRB.INF_OR_UNBD:
            print("Model is either infeasible or unbounded")
        else:
            print(f"Optimization status: {status}")

