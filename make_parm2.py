
from shutil import copyfile
import random
import subprocess
from gridData import Grid
from os import listdir, getcwd, path
import numpy as np

class mol_grid:
    def __init__(self, alive=True,grid_addr=''):
        self.grid_addr = grid_addr.split('/')[0]
        self.alive = alive
        self.lig_grid = None
        self.recep_grid = None
        if alive == True:
            self.lig_grid , self.recep_grid= self.load_grids()
            
        if 'POS' in self.grid_addr:
            self.label = 0
        elif 'NEG' in self.grid_addr:
            self.label = 1

    def load_grids(self):
        lig_grid = Grid(self.grid_addr+'/'+self.grid_addr+'.dx').grid
        recep_grid =  Grid(self.grid_addr+'/receptor.dx').grid
        return lig_grid, recep_grid

    def check_pulse(self):
        if self.alive == True: 
            self.lig_grid = None
            self.recep_grid = None
            self.alive = False
        else: 
            self.lig_grid,self.recep_grid = self.load_grids()
            self.alive = True

class grid_maker:
    def __init__ (self, folder='./',parm0='teste1.in', make_grid=False):
        
        lig_list = [f for f in listdir(folder) if path.isdir(f) and "_ligand_" in f] 
        
        self.make_grid = make_grid
        
        if make_grid:
            self.parm_list =[self.make_parm(parm0,lig) for lig in lig_list]
            print self.parm_list
            for parm in self.parm_list:
                process = subprocess.Popen(['apbs',parm], stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                print stdout
        else:
            self.parm_list = [self.make_parm(parm0,lig) for lig in lig_list]

    def run_apbs(self):
        
        
        for parm in self.parm_list:
            process = subprocess.Popen(['apbs',parm], stdout=subprocess.PIPE)
            stdout = process.communicate()[0]
            
    
    def make_parm(self,parm0,pth):
        with open(parm0) as parm:
            with open(pth+'/'+pth+'.in','w') as new_parm:
                for line in parm:        
                    if "mol pqr lig2.pqr" in line:
                        new_parm.write('    mol pqr '+pth+'/'+pth+'.pqr\n')
                    elif "../receptor.pqr" in line:
                        new_parm.write('    mol pqr '+getcwd()+'/'+'receptor.pqr\n')
                    elif "write pot dx receptor" in line:
                        
                        new_parm.write("    write pot dx "+pth+'/receptor'+'\n')
                    elif "    write pot dx lig2" in line:
                        new_parm.write("    write pot dx "+pth+'/'+pth+'\n')
                    
                    else:
                        new_parm.write(line)
        return pth+'/'+pth+'.in'

    def next_batch(self, size=2):
        
        batch = random.sample(self.parm_list, size)
        batch = [mol_grid(True,parm) for parm in batch]

        b_l = np.array([gd.lig_grid for gd in batch])
        b_r = np.array([gd.recep_grid for gd in batch])
        
        
        
        
        b_sim = np.array([gd.label for gd in batch])
        
        
        return b_l.reshape((-1,25,25,25,1)), b_r.reshape((-1,25,25,25,1)),b_sim.reshape((-1,1))
        
