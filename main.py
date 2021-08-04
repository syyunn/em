import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(654)
# Draw samples from two Gaussian w.p. z_i ~ Bernoulli(phi)
generative_m = np.array([stats.norm(2, 1), stats.norm(5, 1.8)])
z_i = stats.bernoulli(0.75).rvs(100)
x_i = np.array([g.rvs() for g in generative_m[z_i]])

# plot generated data and the latent distributions
x = np.linspace(-5, 12, 150)
plt.figure(figsize=(16, 6))
plt.plot(x, generative_m[0].pdf(x))
plt.plot(x, generative_m[1].pdf(x))
plt.plot(x, generative_m[0].pdf(x) + generative_m[1].pdf(x), lw=1, ls='-.', color='black')
plt.fill_betweenx(generative_m[0].pdf(x), x, alpha=0.1)
plt.fill_betweenx(generative_m[1].pdf(x), x, alpha=0.1)
plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])

plt.show()


if __name__ == "__main__":
    from emalgo import EM

    m = EM(2)
    m.fit(x_i)

    fitted_m = [stats.norm(mu, std) for mu, std in zip(m.mu, m.std)]

    plt.figure(figsize=(16, 6))
    plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
    plt.plot(x, fitted_m[0].pdf(x))
    plt.plot(x, fitted_m[1].pdf(x))
    plt.plot(x, generative_m[0].pdf(x), color='black', lw=1, ls='-.')
    plt.plot(x, generative_m[1].pdf(x), color='black', lw=1, ls='-.')
    plt.show()
    pass
