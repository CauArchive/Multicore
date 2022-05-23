import java.util.concurrent.atomic.*;

public class ex3 {
    public static void main(String[] args) {
        AtomicInteger atmint = new AtomicInteger(0);
        
        Thread t1 = new Thread(new AtomicRunner(atmint));
        Thread t2 = new Thread(new AtomicRunner(atmint));

        System.out.println("======With Atomic=====");
        t1.start();
        t2.start();
        try {
            t1.join();
            t2.join();
        } catch (InterruptedException s) {}
        System.out.println("======================");

        int non_atmint = 0;

        Thread t3 = new Thread(new NonAtomicRunner(non_atmint));
        Thread t4 = new Thread(new NonAtomicRunner(non_atmint));

        System.out.println("====Without Atomic====");
        t3.start();
        t4.start();
        try {
            t3.join();
            t4.join();
        } catch (InterruptedException s) {}
        System.out.println("======================");
    }
}

class AtomicRunner implements Runnable{
    private AtomicInteger atm;
    
    public AtomicRunner(AtomicInteger atmval) {
        this.atm = atmval;
    }

    public void run() {
        for (int i = 0; i < 10; i++) {
            // We can use get(), set(), getAndAdd(), addAndGet()
            int temp = atm.getAndIncrement();
            System.out.println("Atomic: " + temp);
        }
    }
}

class NonAtomicRunner implements Runnable{
    private int atm;
    
    public NonAtomicRunner(int atmval) {
        this.atm = atmval;
    }

    public void run() {
        for (int i = 0; i < 10; i++) {
            int temp = atm++;
            System.out.println("Non-Atomic: " + temp);
        }
    }
}