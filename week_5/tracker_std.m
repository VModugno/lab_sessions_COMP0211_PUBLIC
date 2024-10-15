
function  [H,F_tra]=tracker_std(S_bar,T_bar,Q_hat,Q_bar,R_bar)
    H     = (R_bar + S_bar'*Q_bar*S_bar);
    first = -Q_hat*S_bar;
    second = T_bar'*Q_bar*S_bar;
    F_tra = [first; second];
    %F_tra = [-Q_hat*S_bar ;T_bar'*Q_bar*S_bar];
end