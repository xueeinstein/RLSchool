#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import parl
from parl import layers
from parl.core.fluid.plutils import fetch_value, set_value


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def Q1(self, obs, act):
        return self.critic_model.Q1(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def load(self, ckpt, use_gpu=False, actor_only=False):
        with open(ckpt, 'rb') as f:
            weights_dict = pickle.load(f)

        if actor_only:
            parameters = self.get_actor_params()
        else:
            parameters = self.parameters()

        for var in parameters:
            set_value(var, weights_dict[var], use_gpu)

    def save(self, ckpt):
        weights_dict = dict()
        for var in self.parameters():
            weights_dict[var] = fetch_value(var)

        with open(ckpt, 'wb') as f:
            pickle.dump(weights_dict, f)


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        return self.fc3(hid2)


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

        self.fc4 = layers.fc(size=hid1_size, act='relu')
        self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.fc6 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])
        return Q1, Q2

    def Q1(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        return Q1
