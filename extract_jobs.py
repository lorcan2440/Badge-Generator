with open('careers_list.txt', 'r') as f:
    import random
    all_jobs = f.readlines()
    for _ in range(100):
        choice = random.choice(all_jobs)
        all_jobs.remove(choice)
        print(choice.strip())