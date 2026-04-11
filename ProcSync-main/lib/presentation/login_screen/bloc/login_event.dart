part of 'login_bloc.dart';

abstract class LoginEvent extends Equatable {
  const LoginEvent();

  @override
  List<Object?> get props => [];
}

class LoginInitialEvent extends LoginEvent {}

class GoogleSignInTappedEvent extends LoginEvent {}
