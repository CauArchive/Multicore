import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.Queue;
import java.util.LinkedList;

public class ex1 {

    public static void main(String[] args) {

        System.out.println("====Without BlockingQueue====");
        Queue<Integer> q = new LinkedList<Integer>();

        System.out.println("Insert 5 elemens to the queue");
        for(int i = 0; i < 5; i++) {
            q.add(i);
        }

        for(int i=0;i<6;i++){
            try{
                System.out.println("Remove(In Queue): " + q.remove());
            }catch(Exception e){
                System.out.println("Error: Queue is Already empty");
            }
        }

        System.out.println("=============================");
        
        
        System.out.println("======With BlockingQueue======");
        BlockingQueue bq = new ArrayBlockingQueue<Integer>(5);
        
        System.out.println("Insert 5 elements to the BlockingQueue");
        for(int i = 0; i < 5; i++) {
            try {
                bq.put(i);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        Thread t1 = new InsertThread(bq);

        for(int i = 0; i < 6; i++) {
            try {
                System.out.println("Remove(In BlockingQueue): " + bq.take());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }


        System.out.println("============================");

    }
}

class InsertThread extends Thread {
    private BlockingQueue bq;
    public InsertThread(BlockingQueue bq) {
        this.bq = bq;
        start();
    }
  
    public void run() {
        System.out.println("Insert 1 element to the BlockingQueue After insert 5 elements");
        try {	
			Thread.sleep(2000);
		} catch (InterruptedException e) {
			System.out.println("error");
		}
        try {
            bq.put(5);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
  }