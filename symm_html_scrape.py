#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:31:41 2018

@author: ryanday

Scrape symmetry information from:
    "Crystallographic Space Group Symmetry Tables.html"

"""


class group:
    
    def __init__(self,group_dict):
        
        self.parse_dict(group_dict)
        
        
    def parse_dict(self,group_dict):
        
        self.index = int(group_dict['index'])
        self.num_ops = int(group_dict[' Number of Symmetry Operators '])
        self.name = group_dict[' Space Group Name '].strip()
        self.system = group_dict[' Crystal System '].strip()
        self.laue_class = group_dict[' Laue Class '].strip()
        self.point_group = group_dict[' Point Group '].strip()
        self.lattice_type = group_dict[' Lattice Type '].strip()
        self.symm_ops = group_dict['symmetry']

    def print_summary(self):
        print('Space Group #{:d}: {:s}'.format(self.index,self.name))
        print('Crystal System : {:s}'.format(self.system))
        print('Lattice Type: {:s}'.format(self.lattice_type))
        print('Number of Operations: {:d}'.format(self.num_ops))
        print('Operations:')
        for si in self.symm_ops:
            print(si)
            
    def write_summary(self):
        line = ''
        line+='Space Group #{:d}: {:s}\n'.format(self.index,self.name)
        line+='Crystal System : {:s}\n'.format(self.system)
        line+= 'Lattice Type: {:s}\n'.format(self.lattice_type)
        line+='Number of Operations: {:d}\n'.format(self.num_ops)
        line+='Operations:\n'
        for si in self.symm_ops:
            line+=si+'\n'
        line+='\n\n'
        return line
        
        
              
    
    
    


def read_html(filename):
    groups = []
    new_group_found = False
    tmp_dict = {}
    with open(filename,'r') as origin:
        
        for line in origin:
            line_txt = line.split('<')
            if len(line_txt)>1:
                if line_txt[1]=='H2>':
                    if line_txt[2][:8]=='A NAME =':
                        new_group_found = True
                        index = int(line.split('"')[1])
                        if 'index' in tmp_dict:
                            groups.append(group(tmp_dict))
                        tmp_dict={'index':index,'symmetry':[]}
                        
                if new_group_found:
                    if line_txt[1][:2]=='li':
                        info = line[4:-6].split('=')
                    
                        dtype = info[0]
                        data = ''.join(info[1:])
                        if dtype.strip()!='symmetry':
                            tmp_dict[dtype] = data
                        else:
                            tmp_dict['symmetry'].append(data)
    return groups


def write_groups(filename,groups):
    
    with open(filename,'w') as tofile:
        for g in groups:
            tofile.write(g.write_summary())
    tofile.close()
    
    print('File printed')
                        
if __name__=="__main__":
    
    filename = "/Users/ryanday/Desktop/Crystallographic Space Group Symmetry Tables.html"
    destination = '/Users/ryanday/Documents/UBC/TB_ARPES/space_groups_2.txt'   
    groups = read_html(filename)
    write_groups(destination,groups)
    
                        
    