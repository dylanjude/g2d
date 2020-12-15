#include <chrono> 
#include <stack>
#include <cstdio>
#include <ctime>
#include <string>
#include "timer.h"

using namespace std::chrono; 
using namespace std;

Timer::Timer(){
   counter=0.0;
}

void Timer::tick(){
   this->t0 = high_resolution_clock::now();
}

double Timer::tock(){
   double recent = duration<double>(high_resolution_clock::now() - t0).count();
   counter += recent;
   return recent;
}

// Peek at the elapsed time without stopping the timer
double Timer::peek(){
   double recent = duration<double>(high_resolution_clock::now() - t0).count();
   return recent;
}

string Timer::timestring(){
   std::time_t now = system_clock::to_time_t(std::chrono::system_clock::now());
   std::string s(30, '\0');
   std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
   return s;
}

double Timer::elapsed(){
   return counter;
}

// //
// // Use a stack to allow nested calls of tick() and tock() in the code.
// // For example this should work:
// //
// // timer::tick();
// // for(int i=0; i<3; i++){
// //    timer::tick();
// //    do_something();
// //    printf("iter %d took %f s\n", i, timer::tock());
// //  }
// // printf("the whole loop took %f s\n", timer::tock());

// namespace timer {

//    static stack<time_point<high_resolution_clock>> t0;

//    void tick(){
//       t0.push(high_resolution_clock::now());
//    }

//    double tock(){
//       double t = duration<double>(high_resolution_clock::now() - t0.top()).count();
//       t0.pop();
//       return t;
//    }

//    int empty(){
//       // printf("timer stack size: %d\n",t0.size());
//       return (t0.size()==0);
//    }

//    string timestring(){
//       std::time_t now = system_clock::to_time_t(std::chrono::system_clock::now());
//       std::string s(30, '\0');
//       std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
//       return s;
//    }

// }
