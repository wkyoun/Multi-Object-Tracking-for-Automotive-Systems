classdef n_objectracker
%N_OBJECTRACKER is a class containing functions to track n object in
%clutter. 
%Model structures need to be called:
%sensormodel: a structure specifies the sensor parameters
%           P_D: object detection probability --- scalar
%           lambda_c: average number of clutter measurements per time
%           scan, Poisson distributed --- scalar 
%           pdf_c: clutter (Poisson) intensity --- scalar
%           intensity_c: clutter (Poisson) intensity --- scalar
%motionmodel: a structure specifies the motion model parameters
%           d: object state dimension --- scalar
%           F: function handle return transition/Jacobian matrix
%           f: function handle return predicted object state
%           Q: motion noise covariance matrix
%measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: function handle return transition/Jacobian matrix
%           h: function handle return the observation of the object
%           state 
%           R: measurement noise covariance matrix

properties
    gating      %specify gating parameter
    reduction   %specify hypothesis reduction parameter
    density     %density class handle
end

methods

    function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
        %INITIATOR initializes n_objectrackersei se ela acabou operando hoje ou nao, ou sei la o que class
        %INPUT: density_class_handle: density class handle
        %       P_D: object detection probability
        %       P_G: gating size in decimal --- scalar
        %       m_d: measurement dimension --- scalar
        %       wmin: allowed minimum hypothesis weight --- scalar
        %       merging_threshold: merging threshold --- scalar
        %       M: allowed maximum number of hypotheses --- scalar
        %OUTPUT:  obj.density: density class handle
        %         obj.gating.P_G: gating size in decimal --- scalar
        %         obj.gating.size: gating size --- scalar
        %         obj.reduction.w_min: allowed minimum hypothesis
        %         weight in logarithmic scale --- scalar 
        %         obj.reduction.merging_threshold: merging threshold
        %         --- scalar 
        %         obj.reduction.M: allowed maximum number of hypotheses
        %         used in TOMHT --- scalar 
        obj.density = density_class_handle;
        obj.gating.P_G = P_G;
        obj.gating.size = chi2inv(obj.gating.P_G,m_d);
        obj.reduction.w_min = log(w_min);
        obj.reduction.merging_threshold = merging_threshold;
        obj.reduction.M = M;
    end

    function [estimates_x, estimates_P] = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
        %GNNFILTER tracks n object using global nearest neighbor
        %association 
        %INPUT: obj: an instantiation of n_objectracker class
        %       states: structure array of size (1, number of objects)
        %       with two fields: 
        %                x: object initial state mean --- (object state
        %                dimension) x 1 vector 
        %                P: object initial state covariance --- (object
        %                state dimension) x (object state dimension)
        %                matrix  
        %       Z: cell array of size (total tracking time, 1), each
        %       cell stores measurements of size (measurement
        %       dimension) x (number of measurements at corresponding
        %       time step)  
        %OUTPUT:estimates: cell array of size (total tracking time, 1),
        %       each cell stores estimated object state of size (object
        %       state dimension) x (number of objects)

        % STEPS FOR GNN
        % 1. implement ellipsoidal gating for each predicted local hypothesis seperately, see Note below for details;
        % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
        % 3. find the best assignment matrix using a 2D assignment solver;
        % 4. create new local hypotheses according to the best assignment matrix obtained;
        % 5. extract object state estimates;
        % 6. predict each local hypothesis.
        % 7. perform prediction of each prior
        n_timesteps = size(Z, 1);
        n_objects = size(states, 2);

        estimates = cell(n_timesteps, 1);

        for timestep = 1 : n_timesteps
            z = Z{timestep};
            num_measurements = size(z, 2);
            measurements_select_indexes = false(n_objects, num_measurements);  % (i,j) = true   j th measurement is inside gate for i th target
            for obj_idx = 1 : n_objects
                state_pred = states(obj_idx);
                [~, measurements_select_indexes(obj_idx, :)] = obj.density.ellipsoidalGating(state_pred, z, measmodel, obj.gating.size);
            end
            
            idx_keep = sum(measurements_select_indexes, 1) > 0;               % idx_keep: of size (1) * (num_measments). If idx_keep(i) = 1, the measuremnt falls inside at least one target's gate
            n_gated_measuments = sum(idx_keep);
            measurements_select = z(:, idx_keep);
            measurements_select_indexes = measurements_select_indexes(:, idx_keep);

            % construct 2D cost assignment matrix 
            LossMatrix = inf(n_objects, n_gated_measuments + n_objects);
            for i = 1 : n_objects
                state_pred = states(i);
                for j = find(measurements_select_indexes(i, :))
                    LossMatrix(i, j) = - log(sensormodel.P_D/sensormodel.intensity_c) ...
                    -obj.density.predictedLikelihood(state_pred, measurements_select(:, j) , measmodel);
                end
                LossMatrix(i, i + n_gated_measuments) = - log(1 - sensormodel.P_D);
            end

            [col4row,~,gain] = assign2D(LossMatrix);
            assert(gain~=-1, 'Assignment problem is unfeasible');

            for i = 1 : n_objects
                if col4row(i) <= n_gated_measuments
                    states(:, i) = obj.density.update(states(:, i), measurements_select(:, col4row(i)), measmodel);
                end
            end

            % predict 
            estimates{timestep} = states;
            states = arrayfun(@(s) obj.density.predict(s,motionmodel), states);
        end

        estimates_x = cell(n_timesteps, 1);
        estimates_P = cell(n_timesteps, 1);
        for timestep = 1 : n_timesteps
            for obj_num = 1 : n_objects
                estimates_x{timestep}(:, obj_num) = estimates{timestep}(obj_num).x;
                estimates_P{timestep}(:, :, obj_num) = estimates{timestep}(obj_num).P;
            end
        end
 

    end


    function [estimates_x, estimates_P] = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
        %JPDAFILTER tracks n object using joint probabilistic data
        %association
        %INPUT: obj: an instantiation of n_objectracker class
        %       states: structure array of size (1, number of objects)
        %       with two fields: 
        %                x: object initial state mean --- (object state
        %                dimension) x 1 vector 
        %                P: object initial state covariance --- (object
        %                state dimension) x (object state dimension)
        %                matrix  
        %       Z: cell array of size (total tracking time, 1), each
        %       cell stores measurements of size (measurement
        %       dimension) x (number of measurements at corresponding
        %       time step)  
        %OUTPUT:estimates: cell array of size (total tracking time, 1),
        %       each cell stores estimated object state of size (object
        %       state dimension) x (number of objects)

        % STEPS FOR JPDA
        % 1. implement ellipsoidal gating for each local hypothesis seperately;
        % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
        % 3. find the M best assignment matrices using a M-best 2D assignment solver;
        % 4. normalise the weights of different data association hypotheses;
        % 5. prune assignment matrices that correspond to data association hypotheses with low weights and renormalise the weights;
        % 6. create new local hypotheses for each of the data association results;
        % 7. merge local hypotheses that correspond to the same object by moment matching;
        % 8. extract object state estimates;
        % 9. predict each local hypothesis.

        n_timesteps = size(Z, 1);
        n_objects = size(states, 2);

        estimates = cell(n_timesteps, 1);

        for timestep = 1 : n_timesteps
            z = Z{timestep};
            num_measurements = size(z, 2);
            measurements_select_indexes = false(n_objects, num_measurements);  % (i,j) = true   j th measurement is inside gate for i th target
            for obj_idx = 1 : n_objects
                state_pred = states(obj_idx);
                [~, measurements_select_indexes(obj_idx, :)] = obj.density.ellipsoidalGating(state_pred, z, measmodel, obj.gating.size);
            end
            
            idx_keep = sum(measurements_select_indexes, 1) > 0;               % idx_keep: of size (1) * (num_measments). If idx_keep(i) = 1, the measuremnt falls inside at least one target's gate
            n_gated_measuments = sum(idx_keep);
            measurements_select = z(:, idx_keep);
            measurements_select_indexes = measurements_select_indexes(:, idx_keep);

            % construct 2D cost assignment matrix 
            LossMatrix = inf(n_objects, n_gated_measuments + n_objects);
            for i = 1 : n_objects
                state_pred = states(i);
                for j = find(measurements_select_indexes(i, :))
                    LossMatrix(i, j) = - log(sensormodel.P_D/sensormodel.intensity_c) ...
                    -obj.density.predictedLikelihood(state_pred, measurements_select(:, j) , measmodel);
                end
                LossMatrix(i, i + n_gated_measuments) = - log(1 - sensormodel.P_D);
            end

            [Theta, ~, gainBest]=kBest2DAssign(LossMatrix, obj.reduction.M);
            
            M = length(gainBest);
            weights_log = zeros(M, 1);
            
            % calculate association weights
            for i = 1 : M
                for n = 1 : n_objects
                    which_measurement = Theta(n , i);
                    weights_log(i) = weights_log(i) - LossMatrix(n, which_measurement);
                end
            end

            weights_log = normalizeLogWeights(weights_log);

            % pruning unlikely weight data association events
            hypotheses_indices = 1 : M;  % M hypotheses in total
            [weights_log, hypotheses_indices] = hypothesisReduction.prune(weights_log, hypotheses_indices, obj.reduction.w_min);
            Theta = Theta(:, hypotheses_indices); 
            weights_log = normalizeLogWeights(weights_log);

            betas = zeros(n_objects, n_gated_measuments + 1);  % linear scale probability
            for i = 1 : n_objects
                for i_Theta = 1 : size(Theta, 2)
                    j = Theta(i, i_Theta);   % j=1 means ass. to meas. 1, j>=n_gated_measuments+1 means misdetection
                    if j <= n_gated_measuments
                        betas(i, j) = betas(i, j) + exp(weights_log(i_Theta));
                    else  
                        betas(i, n_gated_measuments + 1) = betas(i, n_gated_measuments + 1) + exp(weights_log(i_Theta));
                    end
                end
            end

            assert(all(sum(betas, 2) - 1 < 0.00001), 'each row should sum 1');
            % sanity check: sum of beta over j = 1 (each row should sum 1)

            % 7. merge local hypotheses that correspond to the same object by moment matching;  
            posterior_states = struct([]);
            for i = 1 : n_objects
                hypotheses = struct([]);
                state_pred = states(i);
                for j = 1 : n_gated_measuments + 1
                    if j ~= n_gated_measuments + 1
                        hypothesis = obj.density.update(state_pred, measurements_select(:, j), measmodel);
                        hypotheses = [hypotheses; hypothesis];
                    else
                        hypothesis = state_pred ;
                        hypotheses = [hypotheses; hypothesis];
                    end
                end
                posterior_states = [posterior_states; obj.density.momentMatching(log(betas(i, :)), hypotheses)];            
            end
            estimates{timestep} = posterior_states;    % 1 * n_objects struct array
            % 9. predict each local hypothesis.
            states = arrayfun(@(s) obj.density.predict(s,motionmodel), posterior_states);
        end

        
        estimates_x = cell(n_timesteps, 1);
        estimates_P = cell(n_timesteps, 1);
        for timestep = 1 : n_timesteps
            for obj_num = 1 : n_objects
                estimates_x{timestep}(:, obj_num) = estimates{timestep}(obj_num).x;
                estimates_P{timestep}(:, :, obj_num) = estimates{timestep}(obj_num).P;
            end
        end


    end 


    function [estimates_x, estimates_P] = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
        %TOMHT tracks n object using track-oriented multi-hypothesis tracking
        %INPUT: obj: an instantiation of n_objectracker class
        %       states: structure array of size (1, number of objects)
        %       with two fields: 
        %                x: object initial state mean --- (object state
        %                dimension) x 1 vector 
        %                P: object initial state covariance --- (object
        %                state dimension) x (object state dimension)
        %                matrix  
        %       Z: cell array of size (total tracking time, 1), each
        %       cell stores measurements of size (measurement
        %       dimension) x (number of measurements at corresponding
        %       time step)  
        %OUTPUT:estimates: cell array of size (total tracking time, 1),
        %       each cell stores estimated object state of size (object
        %       state dimension) x (number of objects)

        
        % STEPS FOR TOMHT
        % 1. for each local hypothesis in each hypothesis tree:
            % 1.1. implement ellipsoidal gating;
        % 2. disconsider measurements that do not fall inside any local hypothesis gate
        % 3. for each local hypothesis in each hypothesis tree:
            % 3.1. calculate missed detection and predicted likelihood for each measurement inside the gate and make sure to save these for future use; 
            % 3.2. create updated local hypotheses and make sure to save how these connects to the old hypotheses and to the new the measurements for future use;
        % 4. for each predicted global hypothesis: 
            % 4.1. create 2D cost matrix; 
            % 4.2. obtain M best assignments using a provided M-best 2D assignment solver; 
            % 4.3. update global hypothesis look-up table according to the M best assignment matrices obtained and use your new local hypotheses indexing;
        % 5. normalise global hypothesis weights and implement hypothesis reduction technique: pruning and capping;
        % 6. prune local hypotheses that are not included in any of the global hypotheses;
        % 7. Re-index global hypothesis look-up table;
        % 8. extract object state estimates from the global hypothesis with the highest weight;
        % 9. predict each local hypothesis in each hypothesis tree.


        
        n_timesteps = size(Z, 1);
        n_objects = size(states, 2);

        estimates = cell(n_timesteps, 1);

        old_global_H.table = ones(1, n_objects);       % initial global hypothesis 
        old_global_H.weights = [log(1)];

        hypotheses_tree = cell(1, n_objects);    % each object has a hypothesis tree
        for i = 1 : n_objects
            hypotheses_tree{i} = states(i); 
        end

        for timestep = 1 : n_timesteps
            z = Z{timestep};
            num_measurements = size(z, 2);
            % 1. create updated hypotheses_tree for all objects
            updated_hypotheses_tree = cell(1, n_objects);
            updated_lik = cell(1, n_objects); % weights{i} matrix (# (measurements + 1) * # (local_hypotheses)
                                                 % last row represents miss
                                                 % detections
            % 2. disregard measurements 
            used_meas = false(num_measurements,1);           %measurement indices inside the gate of undetected objects
            gating_matrix = cell(1, n_objects);
            for i = 1 : n_objects
                %number of local hypotheses in hypothesis tree i
                num_hypo = length(hypotheses_tree{i});
                %construct gating matrix
                gating_matrix{i} = false(num_measurements, num_hypo);
                for j = 1: num_hypo
                    %Perform gating for each local hypothesis
                    [~, gating_matrix{i}(:,j)] = obj.density.ellipsoidalGating(hypotheses_tree{i}(j), z, measmodel, obj.gating.size);
                    used_meas = used_meas | gating_matrix{i}(:,j);
                end
            end
            % 3. for each local hypothesis in each hypothesis tree
            z = z(:, used_meas);
            m = size(z, 2);
            gating_matrix = cellfun(@(x) x(used_meas,:), gating_matrix, 'UniformOutput',false);
            
            new_global_H.table = [];
            new_global_H.weights = [];
            for i = 1 : n_objects
                num_hypo = length(hypotheses_tree{i});
                updated_lik{i} = -inf(m + 1, num_hypo);
                for h = 1 : num_hypo
                    for j = 0 : m
                        hk = (h - 1) * (m + 1) + 1 + j;
                        % not detected
                        if (j == 0)
                            updated_hypotheses_tree{i}(hk, 1) = hypotheses_tree{i}(h);
                            updated_lik{i}(m + 1, h) = log(1 - sensormodel.P_D);
                        else
                            if gating_matrix{i}(j, h)
                                updated_hypotheses_tree{i}(hk, 1) = obj.density.update(hypotheses_tree{i}(h), z(:, j), measmodel);
                                updated_lik{i}(j, h) = log(sensormodel.P_D/sensormodel.intensity_c) + obj.density.predictedLikelihood(hypotheses_tree{i}(h), z(:, j), measmodel);
                            end
                        end
                    end
                end
                
            end

            
            
            % TO-MHT look-up table update
            hk = 0;
            num_global_hypothesis = size(old_global_H.table, 1);
            % cost matrix
            L2 = inf(n_objects);
            L2(logical(eye(n_objects))) = -log(1 - sensormodel.P_D);
            for hkm1 = 1 : num_global_hypothesis
                global_hypothesis = old_global_H.table(hkm1, :);
                prior = old_global_H.weights(hkm1);
                % create cost matrix 
                L1 = inf(n_objects, m); 
                for i = 1 : n_objects
                    hypo_index = global_hypothesis(i);
                    L1(i, :) = -(updated_lik{i}(1:end-1, hypo_index)).';
                end

                L = [L1 L2];
                [col4rowBest, ~, gainBest] = kBest2DAssign(L, ceil(exp(prior)*obj.reduction.M));
                
                Mhkm1 = length(gainBest);
                for i = 1 : Mhkm1
                    hk = hk + 1;
                    for j = 1 : n_objects
                        if col4rowBest(j) > m
                            new_global_H.table(hk, j) = (old_global_H.table(hkm1, j) - 1)  *  (m + 1) + 1;
                        else
                            new_global_H.table(hk, j) = (old_global_H.table(hkm1, j) - 1)  *  (m + 1)  + 1 + col4rowBest(j);
                        end
                    end
                    new_global_H.weights(hk, 1)  = -gainBest(i) + prior; 
                end           
            end

            %[new_global_H.weights, ~] = normalizeLogWeights(new_global_H.weights);

            % prune
            [new_global_H.weights, idx] = ...
                                    hypothesisReduction.prune(new_global_H.weights, 1 : length(new_global_H.weights), obj.reduction.w_min);
            new_global_H.table = new_global_H.table(idx, :);
            % capping
            [new_global_H.weights, idx] = ...
                                    hypothesisReduction.cap(new_global_H.weights, 1 : length(new_global_H.weights), obj.reduction.M);    
            new_global_H.table = new_global_H.table(idx, :);
            [new_global_H.weights, ~] = normalizeLogWeights(new_global_H.weights);
            for i = 1 : n_objects
                temp = new_global_H.table(:,i);
                % prune local hypotheses that are not included in any of the global hypotheses
                updated_hypotheses_tree{i} = updated_hypotheses_tree{i}(unique(temp, 'stable'));
                % reindex global hypothesis look-up table
                [~, ~, new_global_H.table(:, i)] = unique(temp,'rows','stable');
            end




            % sort new_global_H table and hypothesis likelihood
            [~,I] = sort(new_global_H.weights);
            new_global_H.table = new_global_H.table(I, :);
            new_global_H.weights = new_global_H.weights(I);
            % extract highest hypothesis
            for i = 1 : n_objects
                estimates{timestep}(i) =  updated_hypotheses_tree{i}(new_global_H.table(end, i));
            end
            % predict 
            for i = 1 : n_objects
                hypotheses_tree{i} = arrayfun(@(s) obj.density.predict(s,motionmodel), updated_hypotheses_tree{i});
            end
            
        
            old_global_H = new_global_H;
        end




        estimates_x = cell(n_timesteps, 1);
        estimates_P = cell(n_timesteps, 1);
        for timestep = 1 : n_timesteps
            for obj_num = 1 : n_objects
                estimates_x{timestep}(:, obj_num) = estimates{timestep}(obj_num).x;
                estimates_P{timestep}(:, :, obj_num) = estimates{timestep}(obj_num).P;
            end
        end
    end          
end
end




