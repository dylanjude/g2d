#include "g2d.h"
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

int main(int argc, char** argv){

  unordered_map<string,string> inputs;
  ifstream startfile("inputs.g2d");
  string line, name, value;
  double* scratch = new double[1000];
  istringstream iss;

  // set defaults:
  inputs["machs"]   = "0.3";
  inputs["aoas"]    = "2 4 6";
  inputs["reys"]    = "1000000";
  inputs["airfoil"] = "naca0012.xyz";
  inputs["order"]   = "3";
  inputs["eqns"]    = "laminar";

  // parse everything into a map of <string : string>
  printf("# Parsing inputs-------------------------------------\n");
  while (getline(startfile, line)){
    iss = istringstream(line);
    getline(iss, name, ':');
    getline(iss, value, '#');

    transform(name.begin(), name.end(), name.begin(), [](unsigned char c){ return std::tolower(c); });
    name.erase(name.find_last_not_of(" ")+1); // trim trailing spaces
    while(name[0]  == ' ') name.erase(0,1);   // trim leading spaces
    while(value[0] == ' ') value.erase(0,1);  // trim leading spaces

    inputs[name]=value;
    printf("# %16s = %s\n", name.c_str(), value.c_str());
  }
  printf("# ---------------------------------------------------\n");

  int nM,nRey,nAoA;
  int jtot,ktot,order;
  double* machs;
  double* reys;
  double* aoas;
  double* xy;
  int eqns;

  // Spatial order
  order = stoi(inputs["order"]);

  // Mach numbers
  iss = istringstream(inputs["machs"]);
  nM=0;
  while(getline(iss,value,' ')){
    scratch[nM++] = stod(value);
  }
  machs = new double[nM];
  for(int i=0; i<nM; i++){
    machs[i] = scratch[i];
  }

  // AoAs
  iss = istringstream(inputs["aoas"]);
  nAoA=0;
  while(getline(iss,value,' ')){
    scratch[nAoA++] = stod(value);
  }
  aoas = new double[nM];
  for(int i=0; i<nAoA; i++){
    aoas[i] = scratch[i];
  }

  // Reynolds
  iss = istringstream(inputs["reys"]);
  nRey=0;
  while(getline(iss,value,' ')){
    scratch[nRey++] = stod(value);
  }
  reys = new double[nM];
  for(int i=0; i<nRey; i++){
    reys[i] = scratch[i];
  }

  // Read Airfoil Coords
  startfile = ifstream(inputs["airfoil"]);
  printf("# reading [%s]\n",inputs["airfoil"].c_str());
  startfile >> jtot >> ktot;
  printf("# jtot, ktot: %d %d\n", jtot, ktot);
  xy = new double[2*jtot*ktot];
  for(int v=0; v<2; v++){
    for(int k=0; k<ktot; k++){
      for(int j=0; j<jtot; j++){
  	startfile >> xy[(j+k*jtot)*2+v];
      }
    }
  }

  if(inputs["eqns"].compare("euler")==0){
    eqns = EULER;
  } else if(inputs["eqns"].compare("laminar")==0){
    eqns = LAMINAR;
  } else {
    eqns = TURBULENT;
  }

  G2D solver(nM,nRey,nAoA,jtot,ktot,order,machs,reys,aoas,xy,eqns);

  solver.init();

  solver.go();

  solver.write_sols();

  delete[] scratch;
  delete[] machs;
  delete[] aoas;
  delete[] reys;
  delete[] xy;

  return 0;

}
