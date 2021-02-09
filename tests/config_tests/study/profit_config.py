ntrain = 2
variables = {'u': 'Uniform(4.7, 5.3)',
             'v': 'Uniform(0.555, 0.6)',
             'w': 'ActiveLearning(0, 10)',
             'r': 'Independent(0, 1, 0.1)',
             'f': {'kind': 'Output', 'range': 'r'},
             'g': 'Output(r)'}
run = 'python3 ../mockup.py'
