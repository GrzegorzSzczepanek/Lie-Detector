preprocessing_param_combinations = [
    #    Configuration 1
    {
        'lfreq': 0.5,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 2
    {
        'lfreq': 0.5,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 3
    {
        'lfreq': 0.5,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 4
    {
        'lfreq': 0.5,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 5
    {
        'lfreq': 0.5,
        'hfreq': 50,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 6
    {
        'lfreq': 0.5,
        'hfreq': 50,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 7
    {
        'lfreq': 1,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (0.5, 1),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 8
    {
        'lfreq': 1,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (0.5, 1),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 9
    {
        'lfreq': 1,
        'hfreq': 40,
        'notch_filter': [50, 60],
        # 'baseline': (0.5, 1),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 10
    {
        'lfreq': 1,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (0.5, 1),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 11
    {
        'lfreq': 1,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (0, 0.5),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 12
    {
        "lfreq": 2,
        "hfreq": 80,
        "notch_filter": [50, 100],
        "baseline": (-0.1, 0),
        "tmin": -0.1,
        "tmax": 1
    },

    # Configuration 13
    {
        "lfreq": 0.5,
        "hfreq": 100,
        "notch_filter": [50, 100],
        "baseline": (-0.1, 0),
        "tmin": -0.1,
        "tmax": 1
    },

    # Configuration 14
    {
        "lfreq": 4,
        "hfreq": 30,
        "notch_filter": [50, 60],
        "baseline": (0, 0.2),
        "tmin": 0,
        "tmax": 1
    },
    # Configuration 15
    {
        "lfreq": 2,
        "hfreq": 50,
        "notch_filter": [50, 60],
        "baseline": (None, None),
        "tmin": 0,
        "tmax": 0.5
    },
    # Configuration 16
    {
        "lfreq": 13,
        "hfreq": 40,
        "notch_filter": [50, 60],
        "baseline": (0, 0.2),
        "tmin": 0,
        "tmax": 0.8
    },
    # Configuration 17
    {
        "lfreq": 2,
        "hfreq": 40,
        "notch_filter": [50, 60],
        "baseline": (0, 0.5),
        "tmin": 0,
        "tmax": 1
    },

    # Configuration 18
    {
        'lfreq': 2,
        'hfreq': 50,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 19
    {
        'lfreq': 0.5,
        'hfreq': 30,
        'notch_filter': [50, 100],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 20
    {
        'lfreq': 0.5,
        'hfreq': 30,
        'notch_filter': [50, 100],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Add more configurations as needed...
    # Configuration 21
    {
        'lfreq': 3,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 22
    {
        'lfreq': 3,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 23
    {
        'lfreq': 3,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 24
    {
        'lfreq': 3,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 25
    {
        'lfreq': 3,
        'hfreq': 50,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 26
    {
        'lfreq': 3,
        'hfreq': 50,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 27
    {
        'lfreq': 4,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 28
    {
        'lfreq': 4,
        'hfreq': 30,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    # Configuration 29
    {
        'lfreq': 4,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    # Configuration 30
    {
        'lfreq': 4,
        'hfreq': 40,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0.5,
        'tmax': 1
    },
    {
        'lfreq': 0.1,
        'hfreq': 70, 
        'notch_filter': [50, 100],
        'baseline': (-0.2, 0),
        'tmin': -0.2,
        'tmax': 0.8  
    },
    {
        'lfreq': 0.3,
        'hfreq': 45,
        'notch_filter': None,
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 1.2
    },
    {
        'lfreq': 4,
        'hfreq': 13,
        'notch_filter': [50],
        'baseline': (-0.1, 0),
        'tmin': -0.1,
        'tmax': 0.6
    },
    {
        'lfreq': 8,
        'hfreq': 30,
        'notch_filter': [60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.5
    },
    {
        'lfreq': 0.1,
        'hfreq': 100,
        'notch_filter': [50, 100],
        'baseline': (-0.5, -0.1),
        'tmin': -0.5,
        'tmax': 1
    },
    {
        'lfreq': 1,
        'hfreq': 40,
        'notch_filter': None,
        'baseline': (0, 0.2),
        'tmin': 0,
        'tmax': 0.8
    },
    {
        'lfreq': 2,
        'hfreq': 50,
        'notch_filter': [50],
        'baseline': (-0.2, 0),
        'tmin': -0.2,
        'tmax': 0.7
    },
    {
        'lfreq': 0.5,
        'hfreq': 20,
        'notch_filter': [50, 60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.4
    },
    {
        'lfreq': 0.1,
        'hfreq': 13,
        'notch_filter': [50],
        'baseline': (-0.2, 0),
        'tmin': -0.2,
        'tmax': 0.5
    },
    {
        'lfreq': 0.1,
        'hfreq': 45,
        'notch_filter': [50],
        'baseline': (-0.1, 0.1),
        'tmin': -0.1,
        'tmax': 0.9
    },
    {
        'lfreq': 0.3,
        'hfreq': 70,
        'notch_filter': [60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.6
    },
    {
        'lfreq': 0.1,
        'hfreq': 100,
        # 'notch_filter': None,
        'baseline': (-0.3, 0),
        'tmin': -0.3,
        'tmax': 0.8
    },
    {
        'lfreq': 8,
        'hfreq': 13,
        'notch_filter': [50],
        'baseline': (-0.2, 0),
        'tmin': -0.2,
        'tmax': 0.5
    },
    {
        'lfreq': 13,
        'hfreq': 30,
        'notch_filter': [50],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.7
    },
    {
        'lfreq': 0.1,
        'hfreq': 55,
        'notch_filter': [50],
        'baseline': (-0.2, -0.1),
        'tmin': -0.2,
        'tmax': 0.6
    },
    {
        'lfreq': 2,
        'hfreq': 80,
        'notch_filter': [50, 100],
        'baseline': (-0.1, 0),
        'tmin': -0.1,
        'tmax': 1
    },
    {
        'lfreq': 0.5,
        'hfreq': 25,
        'notch_filter': [60],
        'baseline': (None, None),
        'tmin': 0,
        'tmax': 0.3
    },
    {
        'lfreq': 1,
        'hfreq': 35,
        # 'notch_filter': None,
        'baseline': (-0.3, -0.1),
        'tmin': -0.3,
        'tmax': 0.5
    },
    {
        'lfreq': 0.1,
        'hfreq': 15,
        'notch_filter': [50],
        'baseline': (-0.2, 0),
        'tmin': -0.2,
        'tmax': 0.4
    },
    # Configuration A
    # {
    #     'lfreq': 0.2,
    #     'hfreq': 75,
    #     'notch_filter': [50, 60],
    #     'baseline': (-0.1, 0),
    #     'tmin': -0.2,
    #     'tmax': 1.0
    # },
    # Configuration B
    # {
    #     'lfreq': 1.5,
    #     'hfreq': 85,
    #     'notch_filter': [50],
    #     'baseline': (0, 0.1),
    #     'tmin': 0,
    #     'tmax': 0.6
    # },
    # # Configuration C
    # {
    #     'lfreq': 0.4,
    #     'hfreq': 60,
    #     'notch_filter': [50, 100],
    #     'baseline': (None, None),
    #     'tmin': -0.1,
    #     'tmax': 0.7
    # },
    # # Configuration D
    # {
    #     'lfreq': 0.1,
    #     'hfreq': 90,
    #     'notch_filter': [60],
    #     'baseline': (-0.5, -0.1),
    #     'tmin': -0.5,
    #     'tmax': 0.5
    # },
    # # Configuration E
    # {
    #     'lfreq': 2.5,
    #     'hfreq': 45,
    #     'notch_filter': [50],
    #     'baseline': (-0.1, 0.1),
    #     'tmin': -0.2,
    #     'tmax': 0.8
    # },
    # # Configuration F
    # {
    #     'lfreq': 0.3,
    #     'hfreq': 35,
    #     'notch_filter': [50, 60],
    #     'baseline': (0, 0.5),
    #     'tmin': -0.3,
    #     'tmax': 1.0
    # },
    # # Configuration G
    # {
    #     'lfreq': 4.0,
    #     'hfreq': 70,
    #     'notch_filter': [50],
    #     'baseline': (0.1, 0.2),
    #     'tmin': -0.1,
    #     'tmax': 0.6
    # },
    # Configuration H
    # {
    #     'lfreq': 0.2,
    #     'hfreq': 50,
    #     'notch_filter': [50, 100],
    #     'baseline': (0.3, 0.6),
    #     'tmin': 0,
    #     'tmax': 0.9
    # },
    # # Configuration I
    # {
    #     'lfreq': 1.2,
    #     'hfreq': 100,
    #     'notch_filter': [70],
    #     'baseline': (-0.2, 0),
    #     'tmin': -0.2,
    #     'tmax': 0.4
    # },
    # # Configuration J
    # {
    #     'lfreq': 0.5,
    #     'hfreq': 25,
    #     'notch_filter': [60],
    #     'baseline': (0.2, -0.5),
    #     'tmin': 0,
    #     'tmax': 0.5
    # },
]
