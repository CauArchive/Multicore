import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {

    private final ReadWriteLock rwlock = new ReentrantReadWriteLock();
    private final Lock readLock = rwlock.readLock();
    private final Lock writeLock = rwlock.writeLock();
    private final List<Integer> list = new ArrayList<>();

    public void setElem(int value){
        writeLock.lock();
        try {
            list.add(value);
        } finally {
            writeLock.unlock();
        }
    }

    public int getElem(int index){
        readLock.lock();
        try {
            return list.get(index);
        } finally {
            readLock.unlock();
        }
    }

    public static void main(String[] args) {

        ex2 ex2 = new ex2();
        ex2.setElem(1);
        ex2.setElem(2);
        ex2.setElem(3);

        System.out.println(ex2.getElem(2));
    }
}