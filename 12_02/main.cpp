#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <random>
#include <chrono>

class BankAccount {
private:
    double balance;
    std::mutex mtx;
    std::condition_variable cv;

public:
    BankAccount(double initial_balance) : balance(initial_balance) {}

    void withdraw(double amount, const std::string& thread_name) {
        std::unique_lock<std::mutex> lock(mtx);
        while (balance < amount) {
            std::cout << thread_name << " waiting: insufficient funds (balance = " << balance << ", need = " << amount << ")\n";
            cv.wait(lock); 
        }
        balance -= amount;
        std::cout << thread_name << " withdrew " << amount << ", new balance = " << balance << "\n";
        lock.unlock();
        cv.notify_all(); 
    }

    void deposit(double amount, const std::string& thread_name) {
        std::lock_guard<std::mutex> lock(mtx);
        balance += amount;
        std::cout << thread_name << " deposited " << amount << ", new balance = " << balance << "\n";
        cv.notify_all(); 
    }

    double get_balance() {
        std::lock_guard<std::mutex> lock(mtx);
        return balance;
    }
};

void withdraw_task(BankAccount& account, const std::string& name, int operations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(10.0, 50.0);

    for (int i = 0; i < operations; ++i) {
        double amount = dis(gen);
        account.withdraw(amount, name);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
    }
}

void deposit_task(BankAccount& account, const std::string& name, int operations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(20.0, 100.0);

    for (int i = 0; i < operations; ++i) {
        double amount = dis(gen);
        account.deposit(amount, name);
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); 
    }
}

int main() {
    BankAccount account(100.0); 

    std::vector<std::thread> threads;
    threads.emplace_back(withdraw_task, std::ref(account), "Withdrawer-1", 5);
    threads.emplace_back(withdraw_task, std::ref(account), "Withdrawer-2", 5);
    threads.emplace_back(withdraw_task, std::ref(account), "Withdrawer-3", 5);
    threads.emplace_back(deposit_task, std::ref(account), "Depositor-1", 5);
    threads.emplace_back(deposit_task, std::ref(account), "Depositor-2", 5);

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final balance: " << account.get_balance() << "\n";

    return 0;
}