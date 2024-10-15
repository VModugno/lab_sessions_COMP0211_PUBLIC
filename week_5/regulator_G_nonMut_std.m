function  [G]=regulator_G_nonMut_std(obj,S_bar)

 G    = [S_bar; -S_bar; eye(obj.N*obj.m); -eye(obj.N*obj.m)];

end
