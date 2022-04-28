import java.util.*;
import java.lang.*;

// command-line execution example) java pc_static_block 6 200000

// Thread Extended Class for PrimeCheck
class PrimeCheckThread extends Thread {
  int lo;
  int hi;
  int idx;
  int ans = 0;
  long[] times;

  // get low & high
  // thread_idx & exec_times for saving execution time
  PrimeCheckThread(int l, int h, int thread_idx,long[] exec_times) {
    this.lo=l; 
    this.hi=h; 
    this.idx=thread_idx; 
    this.times=exec_times;
  }

  @Override
  public void run() {
    // start time
    long startTime = System.currentTimeMillis();
    for(int i=lo; i<hi; i++){
      if(isPrime(i)) ans++;
    }
    // end time
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

public class pc_static_block
{
  private static int NUM_END = 200000;
  private static int NUNM_THREADS = 20;
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
      threads[i] = new PrimeCheckThread((i*NUM_END/NUNM_THREADS), (i+1)*NUM_END/NUNM_THREADS, i, times);
      threads[i].start();
    }

    try{
      for(int i=0; i<NUNM_THREADS; i++){
        threads[i].join();
        ans += threads[i].ans;
      }
    }catch (InterruptedException IntExp) {
      IntExp.printStackTrace();
    }

    return ans;
  }
}

