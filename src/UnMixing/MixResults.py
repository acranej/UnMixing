from dataclasses import dataclass 
@dataclass(frozen = True)
class MixResults:
    w: float
    mu: float
    sigma: float 
    
    @property
    def meanW(self) -> float:
        return self.w.mean
    
    @property
    def meanMu(self) -> float:
        return self.mu.mean
    
    @property
    def meanSigman(self) -> float:
        return self.sigma.mean
    
    