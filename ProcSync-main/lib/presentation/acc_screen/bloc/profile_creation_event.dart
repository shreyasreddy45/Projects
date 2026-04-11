import 'package:equatable/equatable.dart';

abstract class ProfileCreationEvent extends Equatable {
  const ProfileCreationEvent();

  @override
  List<Object?> get props => [];
}

class ProfileCreationInitialEvent extends ProfileCreationEvent {}

class UsernameChangedEvent extends ProfileCreationEvent {
  final String username;

  const UsernameChangedEvent(this.username);

  @override
  List<Object?> get props => [username];
}

class RoleSelectedEvent extends ProfileCreationEvent {
  final String role;

  const RoleSelectedEvent(this.role);

  @override
  List<Object?> get props => [role];
}

class BranchSelectedEvent extends ProfileCreationEvent {
  final String branch;

  const BranchSelectedEvent(this.branch);

  @override
  List<Object?> get props => [branch];
}

class YearSelectedEvent extends ProfileCreationEvent {
  final String year;

  const YearSelectedEvent(this.year);

  @override
  List<Object?> get props => [year];
}

class PositionSelectedEvent extends ProfileCreationEvent {
  final String position;

  const PositionSelectedEvent(this.position);

  @override
  List<Object?> get props => [position];
}

class CreateProfileEvent extends ProfileCreationEvent {}
