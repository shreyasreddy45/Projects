import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:equatable/equatable.dart';

part 'login_event.dart';
part 'login_state.dart';

class LoginBloc extends Bloc<LoginEvent, LoginState> {
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseAuth _auth = FirebaseAuth.instance;

  LoginBloc() : super(LoginState.initial()) {
    on<LoginInitialEvent>(_onInitialize);
    on<GoogleSignInTappedEvent>(_onGoogleSignInTapped);
  }

  void _onInitialize(LoginInitialEvent event, Emitter<LoginState> emit) {
    emit(state.copyWith(isLoading: false));
  }

  Future<void> _onGoogleSignInTapped(
    GoogleSignInTappedEvent event,
    Emitter<LoginState> emit,
  ) async {
    emit(state.copyWith(isLoading: true, errorMessage: null));

    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) {
        emit(state.copyWith(isLoading: false));
        return;
      }

      final GoogleSignInAuthentication googleAuth =
          await googleUser.authentication;

      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      await _auth.signInWithCredential(credential);

      emit(state.copyWith(isLoading: false, isAuthenticated: true));
    } catch (e) {
      print("Google Sign-In Error: $e");
      emit(state.copyWith(
        isLoading: false,
        errorMessage: 'Google Sign-In failed. Please try again.',
      ));
    }
  }
}
