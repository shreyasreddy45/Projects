part of 'login_bloc.dart';

class LoginState extends Equatable {
  final bool isLoading;
  final bool isAuthenticated;
  final String? errorMessage;

  const LoginState({
    required this.isLoading,
    required this.isAuthenticated,
    this.errorMessage,
  });

  factory LoginState.initial() {
    return const LoginState(
      isLoading: false,
      isAuthenticated: false,
      errorMessage: null,
    );
  }

  LoginState copyWith({
    bool? isLoading,
    bool? isAuthenticated,
    String? errorMessage,
  }) {
    return LoginState(
      isLoading: isLoading ?? this.isLoading,
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
      errorMessage: errorMessage,
    );
  }

  @override
  List<Object?> get props => [isLoading, isAuthenticated, errorMessage];
}
