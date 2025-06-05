import random
import time
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
from game.wrapped_flappy_bird import GameState
from src.model import Model, GameProcessor


class TrainingSession:
    def __init__(self, model_path=None, log_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_or_create_model(model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.loss_fn = nn.MSELoss()
        self.memory = []
        self.game = GameState()
        self.log_path = Path(log_path) if log_path else Path('training_log.csv')
        self.start_epoch = 0
        self.best_score = 0

        # Load previous training state if exists
        if self.log_path.exists():
            log_data = pd.read_csv(self.log_path)
            if not log_data.empty:
                self.start_epoch = log_data['epoch'].max() + 1
                self.best_score = log_data['score'].max()
                print(f"Resuming training from epoch {self.start_epoch}, best score: {self.best_score}")

    def _load_or_create_model(self, model_path):
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.train()
            return model
        print("Creating new model")
        return FlappyBirdAI().to(self.device)

    def _store_experience(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.model.memory_capacity:
            self.memory.pop(0)

    def _get_training_batch(self):
        batch = random.sample(self.memory, min(len(self.memory), self.model.batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)

        return states, actions, rewards, next_states, dones

    def _update_model(self):
        states, actions, rewards, next_states, dones = self._get_training_batch()

        current_q = self.model(states)
        current_q_selected = torch.sum(current_q * actions, dim=1)

        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]

        target_q = rewards.squeeze() + (1 - torch.tensor(dones, dtype=torch.float32).to(
            self.device)) * self.model.discount_factor * next_q

        assert current_q_selected.shape == target_q.shape, f"Shape mismatch: {current_q_selected.shape} vs {target_q.shape}"

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q_selected, target_q)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _save_checkpoint(self, epoch, metrics):
        model_path = f"models/current_model.pth"
        torch.save(self.model, model_path)

        if metrics['score'] > self.best_score:
            self.best_score = metrics['score']
            torch.save(self.model, f"models/best_model_{epoch}.pth")

        self._record_metrics(metrics)

    def _record_metrics(self, metrics):
        df = pd.DataFrame([metrics])
        df.to_csv(self.log_path, mode='a', header=not self.log_path.exists(), index=False)

    def run(self):
        start_time = time.time()
        exploration_schedule = np.linspace(
            self.model.exploration_rate,
            self.model.min_exploration,
            self.model.total_epochs
        )

        if self.start_epoch > 0:
            exploration_schedule = exploration_schedule[self.start_epoch:]

        action = torch.zeros([self.model.action_space], dtype=torch.float32).to(self.device)
        action[0] = 1
        frame, _, _, score = self.game.frame_step(action)
        state = self._prepare_initial_state(frame)

        for epoch in range(self.start_epoch, self.model.total_epochs):
            with torch.no_grad():
                q_values = self.model(state)

            if random.random() <= exploration_schedule[epoch - self.start_epoch]:
                action_idx = torch.randint(self.model.action_space, (1,)).to(self.device)
            else:
                action_idx = torch.argmax(q_values)

            action = torch.zeros([self.model.action_space], dtype=torch.float32).to(self.device)
            action[action_idx] = 1

            next_frame, reward, terminal, score = self.game.frame_step(action)
            next_state = self._prepare_next_state(state, next_frame)

            reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
            self._store_experience((state, action.unsqueeze(0), reward_tensor, next_state, terminal))

            loss = self._update_model()

            state = next_state

            if epoch % 100 == 0 or terminal:
                metrics = {
                    'epoch': epoch,
                    'time_elapsed': time.time() - start_time,
                    'exploration_rate': exploration_schedule[epoch - self.start_epoch],
                    'reward': reward,
                    'max_q': torch.max(q_values).item(),
                    'score': score,
                    'loss': loss
                }
                print(f"Epoch: {epoch:6d} | "
                      f"Score: {score:4d} | "
                      f"Loss: {loss:.4f} | "
                      f"Q: {metrics['max_q']:.2f} | "
                      f"Exploration: {metrics['exploration_rate']:.4f}")

                self._save_checkpoint(epoch, metrics)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        pygame.quit()
        print(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")

    def _prepare_initial_state(self, frame):
        """Подготавливает начальное состояние из 4 одинаковых кадров"""
        processed = GameProcessor.preprocess_frame(frame)
        tensor = GameProcessor.frame_to_tensor(processed)
        return torch.cat([tensor] * 4).unsqueeze(0).to(self.device)

    def _prepare_next_state(self, current_state, new_frame):
        """Обновляет состояние, добавляя новый кадр и удаляя самый старый"""
        processed = GameProcessor.preprocess_frame(new_frame)
        tensor = GameProcessor.frame_to_tensor(processed)
        return torch.cat([current_state.squeeze(0)[1:], tensor]).unsqueeze(0).to(self.device)
