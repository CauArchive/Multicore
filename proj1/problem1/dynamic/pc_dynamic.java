import java.util.*;
import java.lang.*;

// command-line execution example) java pc_static_cyclic 6 200000

public class pc_dynamic
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
    PrimeCheckThread.isPrime = NUNM_THREADS-1;
    PrimeCheckThread[] threads = new PrimeCheckThread[NUNM_THREADS];

    for(int i=0; i<NUNM_THREADS; i++){
      threads[i] = new PrimeCheckThread(i, i);
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

  private static boolean isPrime(int x){
    int i;
    if(x<=1) return false;
    for(i=2; i<x; i++){
      if(x%i==0) return false;
    }
    return true;
  }

  static class PrimeCheckThread extends Thread {
    int ans = 0; 
    int x;
    int num;
    static int isPrime;
  
    PrimeCheckThread(int x, int thread_idx) {
      this.x=x;
      this.num = thread_idx;
    }
  
    @Override
    public void run() {
      long startTime = System.currentTimeMillis();
      
      if(isPrime(this.x)) ans++;
      while(isPrime < NUM_END){
        // If this.x is updated, and another thread make isPrime to over NUM_END,
        // this thread will be terminated without checking isPrime(this.x).
        // So, the Order is important
        // Just right after Update this.x, should run check isPrime(this.x)
        this.x = update();
        if(isPrime(this.x)) ans++;
      }
      long endTime = System.currentTimeMillis();
      times[this.num] = (endTime - startTime);
    }
  
    static synchronized int update(){
      isPrime++;
      return isPrime;
    }
  }
}