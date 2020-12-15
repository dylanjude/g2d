#pragma once
#include <chrono> 
#include <stack>
#include <cstdio>
#include <ctime>
#include <string>

// using namespace std::chrono;
// using namespace std;

class Timer {

protected:
   double counter;
   std::chrono::time_point<std::chrono::high_resolution_clock> t0;

public:
   Timer();
   virtual ~Timer(){};
   virtual void tick();
   virtual double tock();
   virtual double peek();
   static std::string timestring();
   double elapsed();
};
