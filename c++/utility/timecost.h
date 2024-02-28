#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <sstream>

class timecost
{
private:
    struct timeval _start;
    struct timeval _end;
    std::string _name;
    float time_diff(struct timeval *start, struct timeval *end) {
        return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
    }
public:
    timecost(std::string item_name):_name(item_name){
        gettimeofday(&_start, NULL);
    }
    ~timecost(){
        gettimeofday(&_end, NULL);
        float diff = time_diff(&_start, &_end);
        std::cout << _name << "time spent:" << diff << "sec" << std::endl;

        std::ostringstream oss;
        oss<<diff;
        std::string diff_str(oss.str());
        std::string cmd = "echo " + diff_str + " >> ";
        if("cpu" == _name){
            std::string f = cmd + "cpu.txt";
            system(f.c_str());
        }else if("cuda" == _name){
            std::string f = cmd + "cuda.txt";
            system(f.c_str());
        }
    }
};


