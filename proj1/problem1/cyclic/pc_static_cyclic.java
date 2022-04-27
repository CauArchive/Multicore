import java.util.*;
import java.lang.*;

// command-line execution example) java pc_static_cyclic 6 200000

class PrimeCheckThread extends Thread {
  int start; int end; int num;
  int ans = 0; int idx;
  long[] times;

  PrimeCheckThread(int num_of_thread, int start, int end, int thread_idx,long[] exec_times) {
    this.start = start;
    this.end = end;
    this.num = num_of_thread;
    this.idx=thread_idx; 
    this.times=exec_times;
  }

  @Override
  public void run() {
    long startTime = System.currentTimeMillis();
    for(int i=start; i<end; i+=num) {
      if(isPrime(i)) ans++;
    }
    long endTime = System.currentTimeMillis();
    times[idx] = (endTime - startTime);
  }


  private static boolean isPrime(int x){
    int i;
    if(x<=1) return false;
    for(i=2; i<x; i++){
      if(x%i==0) return false;
    }
    return true;
  }
}

public class pc_static_cyclic
{
  private static int NUM_END = 200000;
  private static int NUNM_THREADS = 4;
  private static long[] times;

  public static void main(String[] args){
    if(args.length == 2){
      NUNM_THREADS = Integer.parseInt(args[0]);
      NUM_END = Integer.parseInt(args[1]);
    }
    times = new long[NUNM_THREADS];
    
    int counter=0;
    int i;
    long startTime = System.currentTimeMillis();
    counter = getAns();
    long endTime = System.currentTimeMillis();
    long timeDiff = endTime - startTime;

    for(i=0; i<NUNM_THREADS; i++){
      System.out.println("Thread " + i + ": " + times[i]+ "ms");
    }
    System.out.println("Program Execution Time: " + timeDiff + "ms");
    System.out.println("1..." + (NUM_END-1) + " prime# counter=" + counter);
  }

  private static int getAns(){
    int ans = 0;
    PrimeCheckThread[] threads = new PrimeCheckThread[NUNM_THREADS];

    for(int i=0; i<NUNM_THREADS; i++){
      threads[i] = new PrimeCheckThread(NUNM_THREADS, i, NUM_END, i, times);
      threads[i].start();
    }

    try{
      for(int i=0; i<NUNM_THREADS; i++){
        threads[i].join();
        ans += threads[i].ans;
      }
    }catch (InterruptedException IntExp) {
    }

    return ans;
  }
}

