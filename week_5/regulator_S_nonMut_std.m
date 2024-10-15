function  [S]=regulator_S_nonMut_std(obj,T_bar)

 S  = [-T_bar; T_bar; zeros(obj.N*obj.m,obj.n); zeros(obj.N*obj.m,obj.n)];

end
