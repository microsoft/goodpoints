metric_to_tex = {
    'final_mmd': 'MMD',
    'full_mmd': 'MMD',
    'gold_mmd': 'Post-Burn-in MMD',
    'final_ed': 'Energy Distance',
    'mmd_post_thin': 'Post-Thinning MMD',
    'mse': 'MSE',
}

method_to_tex = {
    'PBI-Compress++': 'Burn-in Oracle + Compress++',
    'PBI-Standard': 'Burn-in Oracle + Standard',
    'PBI-CVX': 'Burn-in Oracle + RT',
    'PBI-CP': 'Burn-in Oracle + CT',
    'GBC-EQ': 'Stein Kernel Thinning',
    'GBC-CVX(r=n**0.4)': 'Stein Recombination ($\\tau={0.4}$)',
    'GBC-CVX(r=n**0.5)': 'Stein Recombination ($\\tau={0.5}$)',
    'GBC-CP(r=n**0.4)': 'Stein Cholesky ($\\tau={0.4}$)',
    'GBC-CP(r=n**0.5)': 'Stein Cholesky ($\\tau={0.5}$)',
    'LBC-EQ(r=n**0.4)': 'Low-rank SKT ($\\tau={0.4}$)',
    'LBC-EQ(r=n**0.5)': 'Low-rank SKT ($\\tau={0.5}$)',
    'LBC-CVX(r=n**0.4)': 'Low-rank SR ($\\tau={0.4}$)',
    'LBC-CVX(r=n**0.5)': 'Low-rank SR ($\\tau={0.5}$)',
    'LBC-CP(r=n**0.4)': 'Low-rank SC ($\\tau={0.4}$)',
    'LBC-CP(r=n**0.5)': 'Low-rank SC ($\\tau={0.5}$)',
    'Standard': 'Standard Thinning',
}

# marker indicates debiasing
method_to_marker = {
    'Stein Thinning': 'o',
    'GBC-EQ': '^',
    'GBC-CVX(r=n**0.4)': '^',
    'GBC-CVX(r=n**0.5)': '^',
    'GBC-CP(r=n**0.4)': '^',
    'GBC-CP(r=n**0.5)': '^',
    'LBC-EQ(r=n**0.4)': 'X',
    'LBC-EQ(r=n**0.5)': 'X',
    'LBC-CVX(r=n**0.4)': 'X',
    'LBC-CVX(r=n**0.5)': 'X',
    'LBC-CP(r=n**0.4)': 'X',
    'LBC-CP(r=n**0.5)': 'X',
    'PBI-Compress++': '*',
    'PBI-Standard': 's',
    'PBI-CVX': '*',
    'PBI-CP': '*',
    'Standard': 's',
}

# linestyle indicates thinning
method_to_ls = {
    'PBI-Compress++': 'dotted',
    'PBI-CVX': 'dotted',
    'PBI-CP': 'dotted',
    'PBI-Standard': 'dotted',
    'Stein Thinning': 'dotted',
    'GBC-EQ': 'dashed',
    'GBC-CVX(r=n**0.4)': (1, (5, 5)),
    'GBC-CVX(r=n**0.5)': (3, (5, 5)),
    'GBC-CP(r=n**0.4)': (1, (5, 5)),
    'GBC-CP(r=n**0.5)': (3, (5, 5)),
    'LBC-EQ(r=n**0.4)': (0, (5, 5)),
    'LBC-EQ(r=n**0.5)': (2, (5, 5)),
    'LBC-CVX(r=n**0.4)': (0, (5, 5)),
    'LBC-CVX(r=n**0.5)': (2, (5, 5)),
    'LBC-CP(r=n**0.4)': (0, (5, 5)),
    'LBC-CP(r=n**0.5)': (2, (5, 5)),
    'Standard': 'dotted',
}

method_to_color = {
    'PBI-Compress++': 'tab:pink',
    'PBI-CVX': 'tab:pink',
    'PBI-CP': 'tab:pink',
    'PBI-Standard': '#1f77b4',
    'Stein Thinning': '#ff7f0e',
    'GBC-EQ': '#2ca02c',
    'GBC-CVX(r=n**0.4)': '#2ca02c',
    'GBC-CVX(r=n**0.5)': '#d62728',
    'GBC-CP(r=n**0.4)': '#2ca02c',
    'GBC-CP(r=n**0.5)': '#d62728',
    'LBC-EQ(r=n**0.4)': '#9467bd',
    'LBC-EQ(r=n**0.5)': '#8c564b',
    'LBC-CVX(r=n**0.4)': '#9467bd',
    'LBC-CVX(r=n**0.5)': '#8c564b',
    'LBC-CP(r=n**0.4)': '#9467bd',
    'LBC-CP(r=n**0.5)': '#8c564b',
   'Standard': '#1f77b4',
}

mcmc_name_to_title = {
    'PRECOND-MALA': 'P-MALA'
}