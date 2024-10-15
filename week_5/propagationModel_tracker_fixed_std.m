

function  [S_bar,S_bar_C,T_bar,T_bar_C,Q_hat,Q_bar,R_bar] = propagationModel_tracker_fixed_std(obj,A,B,C,Q,R)

    for k = 1:obj.N
        for j = 1:k
            S_bar(obj.n*(k-1)+(1:obj.n),obj.m*(k-j)+(1:obj.m))   = A^(j-1)*B;
            S_bar_C(obj.q*(k-1)+(1:obj.q),obj.m*(k-j)+(1:obj.m)) = C*A^(j-1)*B;
        end
        T_bar(obj.n*(k-1)+(1:obj.n),1:obj.n)                     = A^k;
        T_bar_C(obj.q*(k-1)+(1:obj.q),1:obj.n)                   = C*A^k;
        Q_hat(obj.q*(k-1)+(1:obj.q),obj.n*(k-1)+(1:obj.n))       = Q*C;
        Q_bar(obj.n*(k-1)+(1:obj.n),obj.n*(k-1)+(1:obj.n))       = C'*Q*C;
        R_bar(obj.m*(k-1)+(1:obj.m),obj.m*(k-1)+(1:obj.m))       = R;
    end
end