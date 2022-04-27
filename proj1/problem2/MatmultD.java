import java.util.*;
import java.lang.*;

// command-line execution example) java MatmultD 6 < mat500.txt
// 6 means the number of threads to use

// Cyclic
// Thread
class RowMultiplyThread extends Thread {
  private final int[][] result;
  private int[][] matrix1;
  private int[][] matrix2;
  long[] times;

  int start; int end; int num;

  public RowMultiplyThread(int[][] result, int[][] matrix1, int[][] matrix2, int start_idx, int num_of_thread, long[] exec_times){
    this.result = result;
    this.matrix1 = matrix1;
    this.matrix2 = matrix2;
    this.start = start_idx;
    this.num = num_of_thread;
    this.times=exec_times;
  }

  @Override
  public void run(){
    long startTime = System.currentTimeMillis();
    for(int i=start; i<matrix1.length;i+=num){
      for(int j=0; j<matrix2[0].length;j++){
        result[i][j] = 0;
        for(int k=0; k<matrix1[i].length;k++){
          result[i][j] += matrix1[i][k] * matrix2[k][j];
        }
      }
    }
    long endTime = System.currentTimeMillis();
    times[start] = (endTime - startTime);
  }
}


public class MatmultD
{
  private static int NUNM_THREADS;
  private static long[] times;
  private static Scanner sc = new Scanner(System.in);

  public static void main(String [] args)
  {
    if (args.length==1) NUNM_THREADS = Integer.valueOf(args[0]);
    else NUNM_THREADS = 1;
    times = new long[NUNM_THREADS];
        
    int a[][]=readMatrix();
    int b[][]=readMatrix();
    int[][] result = new int[a.length][b[0].length];

    long startTime = System.currentTimeMillis();
    int[][] c=multMatrix(a,b);
    long endTime = System.currentTimeMillis();

    printMatrix(c);

    for(int i=0; i<NUNM_THREADS; i++){
      System.out.println("Thread " + i + ": " + times[i]+ "ms");
    }

    System.out.printf("thread_no: %d\n" , NUNM_THREADS);
    System.out.printf("Calculation Time: %d ms\n" , endTime-startTime);

    System.out.printf("[thread_no]:%2d , [Time]:%4d ms\n", NUNM_THREADS, endTime-startTime);
  }

   public static int[][] readMatrix() {
       int rows = sc.nextInt();
       int cols = sc.nextInt();
       int[][] result = new int[rows][cols];
       for (int i = 0; i < rows; i++) {
           for (int j = 0; j < cols; j++) {
              result[i][j] = sc.nextInt();
           }
       }
       return result;
   }

  public static void printMatrix(int[][] mat) {
  System.out.println("Matrix["+mat.length+"]["+mat[0].length+"]");
    int rows = mat.length;
    int columns = mat[0].length;
    int sum = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        System.out.printf("%4d " , mat[i][j]);
        sum+=mat[i][j];
      }
      System.out.println();
    }
    System.out.println();
    System.out.println("Matrix Sum = " + sum + "\n");
  }

  public static int[][] multMatrix(int a[][], int b[][]){//a[m][n], b[n][p]
    if(a.length == 0) return new int[0][0];
    if(a[0].length != b.length) return null; //invalid dims

    int n = a[0].length;
    int m = a.length;
    int p = b[0].length;
    int ans[][] = new int[m][p];

    RowMultiplyThread[] threads = new RowMultiplyThread[NUNM_THREADS];

    for(int i=0; i<NUNM_THREADS; i++){
      threads[i] = new RowMultiplyThread(ans, a, b, i, NUNM_THREADS, times);
      threads[i].start();
    }

    try{
      for(int i=0; i<NUNM_THREADS; i++){
        threads[i].join();
      }
    }catch (InterruptedException IntExp) {
    }

    return ans;
  }
}