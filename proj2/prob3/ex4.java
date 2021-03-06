import java.util.concurrent.*;

public class ex4 {
    public static void main(String[] args) {
        Runnable barrier1Action = new Runnable() {
            public void run() {
                System.out.println("BarrierAction 1 executed ");
            }
        };

        Runnable barrier2Action = new Runnable() {
            public void run() {
                System.out.println("BarrierAction 2 executed ");
            }
        };

        CyclicBarrier barrier1 = new CyclicBarrier(4, barrier1Action);
        CyclicBarrier barrier2 = new CyclicBarrier(4, barrier2Action);

        CyclicBarrierRunnable barrierRunnable1 =
                new CyclicBarrierRunnable(barrier1, barrier2);

        CyclicBarrierRunnable barrierRunnable2 =
                new CyclicBarrierRunnable(barrier1, barrier2);

        CyclicBarrierRunnable barrierRunnable3 =
                new CyclicBarrierRunnable(barrier1, barrier2);

        new Thread(barrierRunnable1).start();
        new Thread(barrierRunnable2).start();
        new Thread(barrierRunnable3).start();

        System.out.println("Run New Thread to finish the program after 3 seconds");
        try {	
			Thread.sleep(3000);
		} catch (InterruptedException e) {
			System.out.println("Interrupted");
		}

        CyclicBarrierRunnable barrierRunnable4 =
        new CyclicBarrierRunnable(barrier1, barrier2);

        new Thread(barrierRunnable4).start();
    }
}

class CyclicBarrierRunnable implements Runnable{

    CyclicBarrier barrier1 = null;
    CyclicBarrier barrier2 = null;

    public CyclicBarrierRunnable(
            CyclicBarrier barrier1,
            CyclicBarrier barrier2) {

        this.barrier1 = barrier1;
        this.barrier2 = barrier2;
    }

    public void run() {
        try {
            Thread.sleep(1000);
            System.out.println(Thread.currentThread().getName() +
                                " waiting at barrier 1");
            this.barrier1.await();

            Thread.sleep(1000);
            System.out.println(Thread.currentThread().getName() +
                                " waiting at barrier 2");
            this.barrier2.await();

            System.out.println(Thread.currentThread().getName() +
                                " done!");

        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        }
    }
}